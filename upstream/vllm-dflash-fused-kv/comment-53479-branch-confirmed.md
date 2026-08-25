Ran `b99d152` on the GB10 rig. **Both predicted cells land.**

| prompt | base | branch | you predicted |
|---|---|---|---|
| 7,292 (unaligned) | request **3**, 4,800 | request **2**, 4,800 | request 2 |
| 6,400 (aligned) | request **3**, 3,200 | request **2**, 3,200 | request 2 |

Default install, retention untouched, cold cache and fresh server per arm.

Carve on send 1 — back-off gone, and only the stops you intended:

```
base    last_cache=4800  -> [4800, 7292]
branch  last_cache=6400  -> [4800, 6400, 7292]     (no stop at 1600/3200)
```

So the retention-aware stop closes the "cost without benefit" case on this runtime too. The hit
moves earlier because base send 2 has the attention hit with **no** Mamba state
(`FullAttention=4800 Mamba=0`); the branch has both at send 2.

Three caveats:

- **Block is 1,600 here, not the 1,648 I reported earlier** — hence 4,800/3,200 rather than
  4,944/3,296. That is your harness's geometry, so these are your cells unscaled. I can't
  reproduce the 1,648 config (both candidate checkpoints resolve to 1,600); treat my earlier
  1,648 figures as an unidentified configuration.
- `usage.prompt_tokens_details.cached_tokens` is **inert** on this build (0 even on confirmed
  hits) — it made my first pass read "never hit" on both arms. Numbers above are
  `prefix_cache_hits_total` deltas, cross-checked against instrumented hit logging.
- Tree is `19c935190` (#52816 branch tip; one commit off `8d6b1832` each way, neither touching
  these paths) — correcting my earlier "dev912" shorthand. Port is verbatim except
  `use_internal_checkpoint`, constantly `False` without #52789, so that path stays untested.

_Disclosure: AI-assisted analysis (Claude Code); I reviewed the traces and arithmetic myself._
