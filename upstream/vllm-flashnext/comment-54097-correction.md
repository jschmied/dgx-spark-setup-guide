@Chinmayrawat15 — thank you, and you are right on the part that matters. Correcting my report:

**The startup claim does not hold on main.** I checked `video_decoders/__init__.py` and the backend
import is lazy inside `decode_video` (`import_module(f".{backend}", __name__)`), exactly as you
describe, so `import vllm.multimodal` completes with a broken torchcodec installed. My "every vLLM
server on such a host dies at startup" was measured on the older preview build
(`0.1.dev20073+g8e685d198`, the `vllm/vllm-openai:qwen38-flash-next` image), where the import chain
is eager. It should have been scoped to that build rather than stated generally — the blast radius
on main is video requests, not the server.

**And your second point kills my suggested fix.** `check_torchcodec_available()` never returns a
bool, it only raises, so widening its `except` to include `OSError` would not give a capability
probe that reports "unavailable" — it would just re-raise a different exception. I proposed a
one-line change without reading what the function actually does with its result.

So the accurate residue of this issue is narrower than the title:

1. On builds where the multimodal import chain is eager, an **installed-but-unloadable** torchcodec
   is fatal at startup where an **absent** one is handled. That asymmetry is the real bug and it is
   worth keeping, because "uninstall the package to fix the server" is a deeply unintuitive remedy.
2. On main it degrades to video requests failing with an `OSError` that the surrounding guard was
   written to catch — `except (ImportError, RuntimeError)` in
   `video_decoders/torchcodec.py:18` — so the guard is still incomplete, just less consequential.

Happy to retitle this to something like *"unloadable torchcodec raises OSError past the
(ImportError, RuntimeError) guards"* and rewrite the body around your findings, with the
startup-vs-video-request distinction stated up front — or to close it in favour of a narrower issue
if you would rather write that one, since you have the reproducer on current main and I do not.

Whichever you prefer. Practical note for anyone landing here meanwhile: on a host without system
ffmpeg, **do not install torchcodec** — absent is handled, broken is not.
