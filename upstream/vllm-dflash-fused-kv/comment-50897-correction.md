Correcting my retracted comment: the branch **does** work here, and my earlier null was a broken
harness — a `cp -a` venv keeps absolute shebangs, so `bin/vllm` ran the original venv's interpreter
and every arm executed unpatched code.

Re-measured with a log line inside `_mamba_block_aligned_split`, so each arm proves which tree ran
and what it computed. GB10 / sm_121a, `Qwen3.8-27B-NVFP4` (hybrid GDN, `align`), MTP n=3, block 1600.

| arm | 6 400 tok (4 blocks exactly) | 7 100 tok | `last_cache_position` |
|---|---:|---:|---:|
| merge-base | 3 200 | 4 800 | 4 800 |
| **this PR** | **4 800** | **6 400** | **6 400** |
| this PR + `--prefix-match-unit 100` | 4 800 | 6 400 | 6 400 |

One full block recovered on both prompts, and warm prefill on the 7 100-token prompt drops from
1.0 s to 0.36 s. `prefix_match_unit` adds nothing on top. The aligned prompt still needs a third
request before it hits — that part looks separate from this PR.
