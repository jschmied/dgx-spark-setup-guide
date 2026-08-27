Heads-up so you don't spend a rebase on this: your source change landed as #53435 (`a9a17e7`).
The two are functionally identical — the merged version is your fix minus the type annotation
and the explanatory comment.

Your **test is not redundant**, though. The merged one instantiates a model and checks
`isinstance(model.layers[0], DFlash2Qwen3DecoderLayer)`; yours is a cheap static guard that also
asserts

```python
assert "DFlashQwen3DecoderLayer(" not in src
```

which targets the exact failure mode that caused this regression — someone re-hardcoding the
class, as #52560 did. That tripwire has no equivalent on main and needs no model build.

Suggestion: close this and reopen just the test plus the `decoder_layer_cls: type[nn.Module]`
annotation and comment as a small follow-up. It won't conflict, and it is the part that would
stop this from regressing a third time.

(Independent GB10 verification that the merged fix works, plus the still-open quantized-drafter
gap behind it: #53122.)
