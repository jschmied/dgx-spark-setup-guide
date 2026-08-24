# Hybrid prefix-cache PRs, measured on GB10

Qwen3.8-27B-NVFP4 (hybrid GDN, `mamba_cache_mode=align`), **MTP `num_speculative_tokens: 3`**,
`--kv-cache-dtype fp8`, chunked prefill at 8192, block size 1600. Prompts calibrated through
`/tokenize` + `/detokenize` to exact token counts, sent three times each, with
`prefix_cache_{queries,hits}_total` read around every single request.

| arm | 6 400 tok (exactly 4 blocks) | 7 100 tok | `last_cache_position` |
|---|---:|---:|---:|
| #50897 merge-base | 3 200 | 4 800 | 4 800 |
| **#50897** | **4 800** | **6 400** | **6 400** |
| #50897 + `--prefix-match-unit 100` | 4 800 | 6 400 | 6 400 |
| #52244 merge-base | 3 200 | 4 800 | 4 800 |
| #52244 | 3 200 | 4 800 | 4 800 |

#50897 recovers one full block on both prompts; warm prefill on the 7 100-token prompt drops from
1.0 s to 0.36 s. `prefix_match_unit` adds nothing on top of it. #52244 leaves these numbers where
they are, which its own `last_cache_position` explains: that PR does not remove the back-off.

## Why `last_cache_position` is in the table

An earlier round of this comparison was **invalid** and had to be retracted upstream. The venv used
for the arms was made with `cp -a`, and a copied venv keeps **absolute shebangs**: its
`bin/vllm` started the *original* venv's interpreter, so every arm silently ran production code and
all of them agreed. Comparing the installed file against the source tree does not catch this --
the copy is correct, it just is not the one being imported.

What catches it is a log line inside the code under test. Each arm here writes
`SCHEDPROBE[<tag>] start=... last_cache=...` from inside `_mamba_block_aligned_split`, so the
journal proves both *which* tree ran and *what it computed*. Read it **after** the probe: before
the first request the path has not executed yet.
