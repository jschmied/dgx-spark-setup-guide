@jhsmith409's instinct about `uniproc_executor` looks right, and there is a diagnosis and a
workaround for it — **credit to [dolf3131](https://github.com/dolf3131/qwen3.8-flash-next-dgx-spark),
who found this independently** and documented it; I am only relaying it here because two people
are blocked on something already solved.

`spawn_ple_offload()` and `wait_ple_offload_ready()` are called from
`vllm/v1/executor/multiproc_executor.py` and **nowhere else**. `uniproc_executor.py` has no such
call, and vLLM selects the uniproc executor by default at TP=1 — so with
`VLLM_PLE_CPU_OFFLOAD=1` the offload worker is **never spawned**, and the GPU side then waits
forever on a peer that does not exist.

That fits every symptom in this thread: the untimed `Queue.get`, `VLLM_PLE_OFFLOAD_READY_TIMEOUT`
not bounding anything (neither wait is on the path it guards), host RAM never rising, and the
hang being identical on sm_121/aarch64/unified and sm_120/x86/discrete — an executor-selection
bug is hardware-independent by construction.

**Workaround:** `--distributed-executor-backend mp`, which forces the multiproc executor at one
GPU.

**Diagnostic, settles it in seconds:**

```bash
docker exec <container> ps -eo pid,rss,comm
# no PleOffloadWorker => it was never spawned
```

Their note adds that they also saw **no disk I/O at all** during the hang and that the announced
IPC socket never appeared in `/tmp` — both consistent with a worker that does not exist rather
than one that is slow.

**Scope of my own claim:** I have not yet reproduced this myself — same image
(`sha256:fc120ece0a38`, `0.1.dev20073+g8e685d198`) and a GB10, but my run is still pending. I am
posting now rather than after because the information unblocks people today. I will follow up
with a confirmation or a correction.

_Disclosure: AI-assisted analysis (Claude Code); the diagnosis relayed here is dolf3131's, not mine._
