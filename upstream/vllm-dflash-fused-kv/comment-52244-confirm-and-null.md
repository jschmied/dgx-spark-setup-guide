Independent confirmation of the defect on different hardware, and a null result for the fix — the
two are worth separating.

**The defect reproduces.** GB10 / sm_121a, `RadixArk/Qwen3.8-27B-NVFP4` (hybrid GDN, `align`), MTP
`num_speculative_tokens: 3`, block size 1600. A prompt of exactly 6 400 tokens — four whole blocks,
calibrated through `/tokenize` + `/detokenize` — caches **3 200**, where the back-off alone predicts
4 800. A 7 100-token prompt caches 4 800, which the back-off predicts exactly. So the extra block is
lost precisely when `prompt_len` is a multiple of the unit, which is your `1072 → 0` row on a
different model, a different drafter and a different page size. It also reproduces with DFlash2 at
block 1648 on our production build, so it is not MTP-specific.

**The fix did not change it here.** Your branch (`62cbf3425`) against its own merge-base, Python
overlaid on one precompiled venv so only the source differs, each arm verified with `cmp` against
the tree:

| arm | 6 400 tok | 7 100 tok |
|---|---:|---:|
| merge-base | 3 200 | 4 800 |
| this PR | 3 200 | 4 800 |
| this PR + `--prefix-match-unit 100` | 3 200 | 4 800 |
| *control: no speculative config* | **6 272** | **6 272** |

The control is why I report the null: with the drafter removed the same prompts reach 4 of 4 blocks,
so the probe resolves a difference when there is one.

**On @ZJY0516's TTFT concern** — the extra chunk did not show up as a cost here either: cold prefill
was 3.14 s on the merge-base and 3.08 s with the PR, which is noise. That is a null on both sides,
so it neither confirms nor clears the objection; it only says the extra chunk is not expensive at
this geometry.

Given the branch is ten days old and conflicting with main, the null may simply be that. Happy to
re-run after a rebase — the harness is scripted, two minutes per arm.
