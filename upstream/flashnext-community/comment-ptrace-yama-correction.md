**Correcting my own fix above: the cause is not Docker seccomp, it is `yama` — and bare metal needs
a different fix.**

I said `--cap-add=SYS_PTRACE` was needed because Docker's default seccomp denies `pidfd_getfd`.
The flag does work, but the reason was wrong, and I only found out by building the same thing
**bare metal**, where it failed identically:

```
RuntimeError: pidfd_getfd: Operation not permitted
  torch/multiprocessing/reductions.py:179 in rebuild_cuda_tensor
  vllm/v1/ple_offload/worker.py:482 in accept_registrations
```

The actual gate is **`kernel.yama.ptrace_scope = 1`**, the default on Ubuntu and on DGX OS:

```bash
$ cat /proc/sys/kernel/yama/ptrace_scope
1
```

`ptrace_scope=1` restricts `PTRACE_MODE_ATTACH` — which `pidfd_getfd` requires — to **descendants
only**. `PleOffloadWorker` and the GPU worker are *siblings*, both children of the engine, so
neither may attach to the other, and the CUDA-IPC tensor handoff is refused. `CAP_SYS_PTRACE`
bypasses yama entirely, which is why the Docker flag fixed it; the container's seccomp profile
was never the obstacle.

So the fix depends on how you run it:

| deployment | fix |
|---|---|
| Docker | `--cap-add=SYS_PTRACE` (as before — right flag, wrong reason) |
| bare metal / systemd | `AmbientCapabilities=CAP_SYS_PTRACE` on the unit |
| bare metal / shell | inherits your login's caps; usually already works |

For systemd specifically, having `cap_sys_ptrace` in `CapabilityBoundingSet` is **not enough** —
that only bounds what may be held. A `User=`/`--uid=` service holds no effective capabilities
unless you also set `AmbientCapabilities`:

```ini
[Service]
AmbientCapabilities=CAP_SYS_PTRACE
```

`sysctl -w kernel.yama.ptrace_scope=0` also works but weakens ptrace restrictions machine-wide;
the ambient capability is scoped to the one service and is the better answer.

Verified end-to-end on a clean bare-metal venv: same checkpoint, same flags, **17.0–17.1 tok/s**,
matching the container result (17.1–17.3) — so the environment reproduces exactly once the
capability is granted.

Apologies for the noise; the flag I gave you works, but if anyone had hit this outside Docker my
explanation would have sent them looking at seccomp for nothing.
