**Title:** The DFlash2 restriction behind the BF16 lm_head change was removed upstream a day earlier

---

The 2026-08-22 notice gives the reason for reverting `lm_head` to BF16 as *"unblocking DFlash2/DSpark drafters"*. That restriction is gone: [vllm#52816](https://github.com/vllm-project/vllm/pull/52816) merged on **2026-08-21**, and the DFlash2 candidate selector now goes through `LogitsProcessor.get_top_k_tokens(lm_head, …)` → `quant_method.apply()` instead of reaching for raw head weights. A quantized head drafts on stock vLLM.

Two data points that it is also quality-neutral, not just loadable:

- The PR author measured GSM8K (1319 questions, 5-shot, greedy) with only the drafter toggled: an **NVFP4 W4A4 head went 0.967 → 0.970** with DFlash2, acceptance 5.80 — same as the bf16-head row.
- On a GB10 (`sm_121a`, TP=1) I ran three drafters against **this checkpoint at revision `554ebba`**, i.e. with the NVFP4 head, 24 replayed agent contexts: BF16 drafter 41.3 tok/s / acceptance 4.24 · INT8 W8A16 43.2 / 4.16 · FP8 W8A8 43.8 / 4.24. All of it drafted through the quantized head without a patch.

For DSpark the target's head should not be the blocker either — that drafter builds its own `ParallelLMHead` (`qwen3_dspark.py`) rather than using the target's.

**What the change costs.** I have not measured the new revision, but the equivalent head swap is measured on the same body, same session, in a map I maintain — column *W4A4 CUTLASS / MLP, static act.*, only the head row differs:

| head | Δtop-1 vs BF16 | MTP n=3 | DFlash2 n=7 |
|---|---:|---:|---:|
| NVFP4 · A4 (`554ebba`) | −2.19 pp | **30.1 tok/s** | **40.3 tok/s** |
| BF16 · A16 | −1.96 pp | 21.7 tok/s | 36.2 tok/s |

So roughly −28 % decode under MTP and −10 % under DFlash2, for +0.23 pp of fidelity — plus the 1.7 GB. Your own post-change GSM8K also reads lower (96.36 % against 97.27 %), which is the opposite of what a wider head should do, so that figure may be worth a second look independently of the rest.

None of this is a request to revert. It is one day of information you could not have had when the change was made.
