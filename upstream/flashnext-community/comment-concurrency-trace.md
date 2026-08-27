**Concurrency numbers, since this thread has only single-stream ones — including mine.**

Traced on one GB10, RadixArk NVFP4, `VLLM_PLE_CPU_OFFLOAD=1`, no speculative decoding, 8k ctx.
Nothing instrumented — `/proc` and `/metrics` counters only, so the measurement does not perturb
what it reports.

| c | aggregate tok/s | per stream | PLE worker cpu% | majflt/token | TTFT | queue |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.1 | 17.1 | 4.9 | 16.0 | 0.22 | 0.00 |
| 8 | 87.5 | 10.9 | 6.1 | 7.0 | 0.53 | 0.00 |
| 16 | 131.6 | 8.2 | 23.9 | 9.6 | 0.83 | 0.00 |
| 32 | 212.0 | 6.6 | 19.9 | 4.3 | 1.19 | 0.00 |
| **48** | **266.8** | 5.6 | 17.9 | 3.6 | **1.60** | 0.01 |

Two results I did not expect:

**1. The PLE CPU offload is nowhere near the bottleneck** — 5-24% of one core across c=1..96. The
gather is cheap because the table is a lookup touching a handful of rows per token, not a GEMM.

**2. Swap cost per token FALLS 4.4x with concurrency** — 16.0 major faults/token at c=1 down to
3.6 at c=48. Batched tokens share n-gram rows and the page cache keeps the hot set, so the
marginal token is far cheaper than the first. The paged table is an argument *for* running this
model concurrent, not a caution against it. Given how much of this thread is about making the
table fit, that seemed worth reporting.

**Every wait I could observe is a queueing wait**, governed entirely by `--max-num-seqs`. With an
adequate cap, `queue_sum` is 0.00 at every concurrency the box holds.

**A confound in my own first attempt, since it is the cheapest thing here to get wrong.** My first
sweep showed throughput flat at ~33 tok/s and I nearly reported a ceiling. It was
`--max-num-seqs 2`, carried over from first-boot testing — at c=4 and c=8 most requests were
queued by my own config. Saturation and a request cap are **indistinguishable in throughput
alone**; `vllm:request_queue_time_seconds_sum` separates them instantly (it hit 142 s while tok/s
stayed flat). If you quote an aggregate number, check `max-num-seqs` first — the default I shipped
costs 4x at c=8.

On **single-stream** I am behind all of you (17.1, no speculation, against 22 / 27 / 28.2 / 31-50).
That axis is where MTP and the HashK work pay off and I have not pulled that lever yet. But on
aggregate the ordering inverts by an order of magnitude, nobody here has published it, and
llama.cpp builds cannot produce it at all (`--parallel 1`) — so it seemed worth putting in the
thread rather than only in my own repo.

Method and harness: https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/load-and-waits.md
