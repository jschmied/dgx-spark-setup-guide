The DFlash2 hunk is correct and minimal — it restores `decoder_layer_cls` both on the class and at
the call site, which is exactly what #53428 needs. I confirmed those are the only two spots #52560
touched there; its other two changes to that `__init__` are behavioural fallbacks, not part of the
regression.

Would you consider splitting the DFlash2 fix out? It currently rides with a CPU sampler seeding
change and an optional-uvloop fallback in the API server worker, neither related to #53428. DFlash2
checkpoints are unloadable on main until this lands, so the two-line fix would move faster on its
own clock — and the uvloop fallback silently swaps the serving event loop, which deserves its own
review rather than a PR titled "Dflash2 load fix".

Heads-up that lint will fail as it stands: 4 trailing-whitespace lines in the new tests and 3 lines
over 88 chars, two of them in the uvloop hunk.
