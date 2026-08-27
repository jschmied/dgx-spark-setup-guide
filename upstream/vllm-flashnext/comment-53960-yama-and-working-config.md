**Working single-GB10 configuration, plus a second permission gate this thread has not hit yet.**

Confirming @jahnclawdmonet's swap point with a different checkpoint, and adding a failure that
bites *after* you get past the OOM.

**Config that serves:** `RadixArk/Qwen3.8-Flash-Next-NVFP4` (125.9 GiB, PLE 47.7 GiB in FP8),
TP=1, `VLLM_PLE_CPU_OFFLOAD=1`, `--distributed-executor-backend mp`,
`--gpu-memory-utilization 0.90`, 8192 ctx, 64 GiB swapfile.

```
weights resident   76.61 GiB   (81.38 with the MTP draft head)
KV cache           30.99 GiB
free at startup   114.3 / 121.63 GiB
```

So the deficit math in this thread is checkpoint-dependent: with an **FP8** PLE at 47.7 GiB
rather than a BF16 one at 95.37, it fits with room to spare and swap only carries the cold
majority of the table. No OOM across ~15 boots today.

**The second gate: `pidfd_getfd: Operation not permitted`.**

Once past the deadlock and the OOM, the run dies ~10 minutes in — *after* both workers load all
206 shards — with only:

```
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

The real error is upstream in the `PleOffloadWorker` stream:

```
RuntimeError: pidfd_getfd: Operation not permitted
  torch/multiprocessing/reductions.py:179 in rebuild_cuda_tensor
  vllm/v1/ple_offload/worker.py:482 in accept_registrations
```

Cause is **`kernel.yama.ptrace_scope=1`**, the default on Ubuntu and DGX OS. It restricts
`PTRACE_MODE_ATTACH` — which `pidfd_getfd` requires — to **descendants only**, and
`PleOffloadWorker` and the GPU worker are *siblings*, both children of the engine. So the
CUDA-IPC tensor handoff in `accept_registrations` is refused.

| deployment | fix |
|---|---|
| Docker | `--cap-add=SYS_PTRACE` |
| bare metal / systemd | `AmbientCapabilities=CAP_SYS_PTRACE` |
| bare metal / shell | inherits the login's caps; usually already works |

For systemd, `cap_sys_ptrace` in `CapabilityBoundingSet` is **not** sufficient — a `User=` service
holds no effective capabilities without `AmbientCapabilities`. `sysctl kernel.yama.ptrace_scope=0`
also works but weakens ptrace machine-wide.

I first reported this as a Docker seccomp restriction and that was wrong; I found out by building
the same stack bare metal, where it failed identically. `CAP_SYS_PTRACE` bypasses yama, which is
why the Docker flag worked for the wrong reason. It does **not** affect mmap-based PLE approaches
(single process, no IPC handoff), which is likely why it has stayed invisible.

Suggestion: since the engine reports `Failed core proc(s): {}` and hides the cause ten minutes
into a load, a preflight check on `VLLM_PLE_CPU_OFFLOAD` — read `/proc/sys/kernel/yama/ptrace_scope`
and warn if it is non-zero and `CAP_SYS_PTRACE` is absent — would save a lot of people that
round-trip. Happy to send a PR if that seems reasonable.

**Working numbers on this box, TP=1**, in case they help anyone size a run — single-stream decode
17.1 tok/s unspeculated, **28.5 with in-checkpoint MTP k=2** (mean accepted length 2.36), and
aggregate 266.8 tok/s at 48 concurrent streams with TTFT 1.6 s.
