@SubSir You are right, and thanks for the correction — I checked it against the merged code: `dflash2/speculator.py` flattens `num_reqs × num_speculative_steps` rows into a single `compute_candidates` call, so the head runs once per step, not once per draft position. My map said the latter; corrected. Its own appendix had it right, so that was an inconsistency on my side, not a measurement problem.

The GSM8K rows are useful — an NVFP4 W4A4 head at 0.967 → 0.970 with the drafter is a cleaner statement of "quality-neutral" than anything I had.
