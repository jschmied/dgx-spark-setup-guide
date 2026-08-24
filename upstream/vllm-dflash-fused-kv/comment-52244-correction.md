Correcting my retracted comment. The harness was broken (a `cp -a` venv keeps absolute shebangs, so
every arm ran unpatched code); re-measured properly, with a log line inside
`_mamba_block_aligned_split` proving which tree ran.

The defect confirmation stands: on GB10, MTP n=3, block 1600, a prompt of exactly 6 400 tokens
caches **3 200** where the back-off alone predicts 4 800 — your `1072 → 0` row on different
hardware.

The fix still does not move it here: 3 200 / 4 800 on both merge-base and branch. The
instrumentation says why — `last_cache_position` stays 4 800 in both, so the back-off is still what
caps this case, and that is not what this PR changes. For contrast, #50897 on the same rig moves
the same two prompts to 4 800 / 6 400.
