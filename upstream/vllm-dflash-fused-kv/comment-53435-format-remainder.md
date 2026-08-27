`ruff check` is clean now — both rules are gone. `ruff format` still wants two changes in
`tests/v1/spec_decode/test_dflash2.py`, though: a blank line after the inline
`from vllm.platforms import current_platform` around line 162, and one trailing blank line at the
end of the file. `ruff format tests/v1/spec_decode/test_dflash2.py` does both.

(Checking these only counts from inside the repo, by the way — run against a copy elsewhere, ruff
does not pick up vLLM's config and reports something unrelated. Cost me a false alarm this morning.)

Note the PR now carries `speculative-decoding` / `qwen` / `dflash` but not `ready`, so
`pre-run-check` still fails and `pre-commit` is still skipped — nothing here has been linted by CI
yet, and won't be until a maintainer applies that label.
