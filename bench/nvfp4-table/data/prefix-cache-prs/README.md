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

## Diverged siblings (#53479's falsifiable predictions)

kamb-code's reading of the identical-repeat numbers above is that they cannot show this PR's gains
on a `0.26.1rc` lookup: the lookup-side drop always caps one block below the deepest state, so
extra store-side depth is invisible to an exact repeat. He predicted movement on **partially**
shared prefixes instead, which is the shape agent traffic actually has.

Parent 6 400 tokens, sibling shares N leading tokens and diverges after. Each pair uses its **own**
corpus region: with one shared parent, the first sibling caches blocks the second one then hits,
which manufactures a base-line hit that is not there (seen, then removed).

| pair | shared | predicted base | predicted with PR | measured base | measured PR |
|---|---:|---:|---:|---:|---:|
| A | 3 400 | 0 | 1 600 | **0** | **0** |
| B | 5 000 | 0 | 3 200 | **3 200** | **3 200** |

Neither prediction holds here: the PR does not move the sibling that should gain, and the base
already hits on the sibling that should not. Both arms carry their own `SCHEDPROBE` line, so the
`last_cache_position` of 4 800 (base) against 6 400 (PR) is on the record for each.

Caveat unchanged: this build predates #52789, so `use_internal_checkpoint` does not exist and the
PR is applied as its intent for that case rather than verbatim.
