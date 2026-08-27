@MaMo7x That error is not this issue — it comes from the KV-cache layout refactor, not the fused
context-KV path.

The message is raised by `CacheConfig.get_resolved_kv_cache_layout` (`vllm/config/cache.py:347` on
main), which throws when `kv_cache_layout` is still `None`. It was introduced by **#51718**
(`8bdc70ec7`, 21 Aug), and the layout is resolved exactly once, in `vllm/v1/engine/core.py:289`.
Worth checking whether the drafter path ends up consulting a `CacheConfig` the engine core never
resolved — that would explain why it only fires when you set `kv_cache_dtype` *inside*
`--speculative-config`.

I cannot reproduce it: my build is `0.26.1rc1.dev912` = main @ `8d6b1832` (18 Aug), which predates
that commit. Yours is `dev913`. So the three days between them contain the change.

`kv_cache_dtype` is a legitimate field of `SpeculativeConfig`, so you are not holding it wrong —
this looks like a genuine regression and belongs on its own issue against #51718 rather than here,
where it will not get the right eyes.
