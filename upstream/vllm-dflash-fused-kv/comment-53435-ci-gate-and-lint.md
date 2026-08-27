The red check here is not your code: `pre-run-check` fails with *"PR must have the 'verified',
'ready', or 'ready-run-all-tests' label to run pre-commit, or the author must have at least 4
merged PRs (found 0)"*. So `pre-commit` was skipped entirely — nothing has actually been linted
yet. A maintainer applying `ready` is what unblocks it.

Worth fixing before that happens, though: after the two merges from main, `ruff` on
`tests/v1/spec_decode/test_dflash2.py` reports two errors, both auto-fixable, plus one formatting
change. `qwen3_dflash.py` is clean and the fix itself is untouched.

```
tests/v1/spec_decode/test_dflash2.py:123:5  I001    Import block is un-sorted or un-formatted
tests/v1/spec_decode/test_dflash2.py:224:5  SIM117  Use a single `with` statement with multiple contexts
```

`ruff check --fix` and `ruff format` on that one file should clear it, so the first CI run after
labelling comes back green instead of costing another round trip.
