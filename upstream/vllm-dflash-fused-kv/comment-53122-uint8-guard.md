One more commit on the branch, from surveying the DFlash2 drafters published since: `f4f218ece`.

`YourHighnessLA/Qwen3.8-27B-DFlash2-NVFP4` packs two 4-bit values per **uint8**, not into int32.
The bit width here is derived from the column count assuming 32-bit containers, and for that layout
the arithmetic returns a clean, wrong answer — 5120 input features over 2560 columns gives
`bits=16, remainder=0`, so the divisibility guard passes and `unpack_from_int32` is handed a uint8
tensor. Only the container dtype separates the two, so it now checks `packed.dtype` and names it in
the error.

Scope note while I was in there: `tcclaviger/Qwen3.8-27B-DFlash2-FP8` is block-scaled fp8 and
stores `weight_scale_inv` rather than `weight_scale`, so it is not supported either — but that one
fails with the explicit error, so it needs no change.
