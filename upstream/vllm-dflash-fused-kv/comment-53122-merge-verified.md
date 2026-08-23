Verified the merge independently: the merge commit left literal conflict markers in
`qwen3_dflash.py`, and after `be2ac329e` cleaned them the resulting `_dequant_kv_slice` is
byte-identical to the branch version — that was the one place a hand-repair could have gone wrong
silently.

Ran my 13 cases against your merged file: all green, and `ruff check` / `ruff format --check` both
clean. Also worth noting for anyone testing this end-to-end: your branch sits on `b389ac294` and
does not carry #52560, so DFlash2 actually loads on it — which is not true of current `main` right
now (#53428).
