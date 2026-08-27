Tested this branch on a second platform and it does not change the numbers for me — reporting it
because the control says the measurement is sensitive, so I suspect I am missing an enabling
condition rather than the change being ineffective.

**Setup.** GB10 / sm_121a, `RadixArk/Qwen3.8-27B-NVFP4` (hybrid GDN, `mamba_cache_mode=align`),
**MTP `num_speculative_tokens: 3`**, `--kv-cache-dtype fp8`, chunked prefill at 8192, block size
1600. This branch (`2b7eaf105`) against its own merge-base, Python overlaid on one precompiled
venv so only the source differs; each arm verifies the installed file against the tree with `cmp`
rather than a keyword. Prompts calibrated through `/tokenize` + `/detokenize` to exact token
counts, each sent three times, `prefix_cache_{queries,hits}_total` read around every request.

| arm | 6 400 tok (exactly 4 blocks) | 7 100 tok | first hit on request |
|---|---:|---:|---:|
| merge-base | 3 200 | 4 800 | 3 |
| **this PR** | 3 200 | 4 800 | 3 |
| **this PR + `--prefix-match-unit 100`** | 3 200 | 4 800 | 3 |
| *control: no speculative config* | **6 272** | **6 272** | **2** |

The control is the reason I trust the two rows above: with the drafter removed the same prompts
reach 4 of 4 blocks instead of 2, the hit arrives one request earlier, and the warm prefill is
0.15 s against 1.42 s. So the probe does resolve a difference when there is one.

Reading `is_eagle_prefix_cache_hashing_enabled`, my config should satisfy it — a speculative config
whose `use_eagle()` is true, prefix caching on, and no `kv_transfer_config`, so it returns
`enable_prefix_caching`. `prefix_match_unit` was accepted (`'prefix_match_unit': 100` appears in the
logged config), yet hits still land only on 1600-token boundaries, so fine-grained matching does not
appear to engage at all here.

Is there a further switch, or does this path require a connector? Happy to re-run — the harness is
scripted and the arms take about two minutes each.
