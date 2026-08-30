### Your current environment

<details>
<summary>Environment</summary>

```
vLLM     : 0.1.dev20073+g8e685d198 (Qwen3.8-Flash-Next preview build, PR #53896)
torch    : 2.13.0+cu130
torchcodec: 0.16.0+cu130
Platform : aarch64, NVIDIA GB10 (DGX Spark, sm_121), CUDA 13.0, Ubuntu 24.04
Note     : NO system ffmpeg installed (libavcodec/libavformat/libavutil absent)
```

Reproduced on a bare-metal venv built from the official
`vllm/vllm-openai:qwen38-flash-next` image's `pip freeze`. The image itself works because it
bundles ffmpeg; the failure appears when the same package set is installed on a host without it.

</details>

### 🐛 Describe the bug

**`torchcodec` is treated as an optional dependency, but an *installed-but-unloadable* one takes
the server down at import.** The guards catch `ImportError` and `RuntimeError`; loading a
`.so` whose dependencies are missing raises **`OSError`**, which escapes.

The perverse consequence: **an absent torchcodec is safe, a broken one is fatal.** Uninstalling
the package fixes the server.

#### Traceback

```
File "vllm/multimodal/__init__.py", line 3, in <module>
File "vllm/multimodal/hasher.py", line 17, in <module>
File "vllm/multimodal/media/__init__.py", line 5, in <module>
File "vllm/multimodal/media/connector.py", line 28, in <module>
File "vllm/multimodal/video.py", line 35, in <module>          # video_decoders/torchcodec.py on main
File "torchcodec/__init__.py", line 13, in <module>
File "torchcodec/decoders/__init__.py", line 7, in <module>
File "torchcodec/_core/__init__.py", line 8, in <module>
File "torchcodec/_core/_metadata.py", line 15, in <module>
File "torchcodec/_core/ops.py", line 35, in <module>
File "torchcodec/_internally_replaced_utils.py", line 71, in load_image_library
File "torch/_ops.py", line 1518, in load_library
    raise OSError(f"Could not load this library: {path}") from e
OSError: Could not load this library: .../torchcodec/libtorchcodec_image.so
```

`vllm/multimodal/__init__.py` imports this chain unconditionally, so **every** vLLM server on such
a host dies at startup — not only video workloads.

#### Two places on `main` (`3e0e1a0`)

`vllm/multimodal/video_decoders/torchcodec.py:18-22`

```python
try:
    from torchcodec.decoders import VideoDecoder
except (ImportError, RuntimeError):        # <- OSError escapes
    VideoDecoder = PlaceholderModule("torchcodec").placeholder_attr(
        "decoders.VideoDecoder"
    )
```

`vllm/utils/import_utils.py:595-608` — `check_torchcodec_available()` likewise catches only
`RuntimeError`, so the capability probe raises instead of returning False.

Note the existing comment there already anticipates the missing-ffmpeg case:

> *torchcodec will raise RuntimeError during import instead of ImportError when system ffmpeg
> unavailable*

That is true when torchcodec's own loader raises. It is **not** the only path: when the failure
surfaces from `torch.ops.load_library`, the exception is `OSError`.

#### Suggested fix

Widen both guards:

```python
except (ImportError, RuntimeError, OSError):
```

This is exactly the widening already done twice for this dependency — #47888 ("Avoid blocking
model launching when no system ffmpeg available for TorchCodec", merged) and #48265 ("Catch
RuntimeError for torchcodec AudioDecoder import"). This issue is the third exception type on the
same code path.

Happy to send the PR if that is welcome.

#### Why it is easy to miss

A broken `.so` is a *host* property, not a package-version one, so CI with ffmpeg present will
never see it. It surfaces on minimal containers and on bare-metal installs that copy a working
image's `pip freeze` onto a host without ffmpeg — which is a common way to reproduce a pinned
environment.

### Before submitting a new issue...

- [x] I searched existing issues and PRs; the two prior widenings (#47888, #48265) cover
  `ImportError` and `RuntimeError` but not `OSError`.
