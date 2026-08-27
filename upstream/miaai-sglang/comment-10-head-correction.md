Correction to my own framing above — I misread which export your table was measured on.

Your README says the canonical table is "DSpark/MTP measured 2026-08-18, DFlash2 2026-08-19,
**all on the packed-FP4-head export**; the default is now the BF16-head twin". So your
50.9 / 25.4 were taken on `RadixArk/Qwen3.8-27B-NVFP4` — the same packed-FP4 head the vLLM
numbers above run on, not the dense BF16 head I assumed.

Two things follow, one in your favour:

1. **The comparison is head-matched**, which makes it cleaner than I claimed, not messier.
   Same body, same quantized `lm_head`, same probe. The remaining differences are the stack and
   the drafter.
2. **My "existence proof" point was wrong** and I withdraw it. You run the quantized head too,
   via the in-place `lm_head.quant_method.apply` selector; there is nothing for vLLM to prove
   there. The `kept eager` fallback discussed in #8 is a separate question from which export
   your numbers came from, and I conflated them.

What stands: the probe, the method, the two vLLM rows, and the caveat that I never ran SGLang
on this box. The BF16-head remark applies only to your current default, not to the table.

Apologies for the noise — the numbers are unchanged, the interpretation was not.
