# 15. SWE-bench: real-issue evaluation with mini-SWE-agent

[← Sampling & variance](14-sampling-and-variance.md) · [Index](README.md)

Pages 12–14 use a bespoke two-task harness (Go concurrency + a Spring/Hibernate stack) to compare models. This page adds a **standardized, third-party** eval: [**SWE-bench**](https://www.swebench.com/) — real GitHub issues, where a model must produce a patch that makes the repository's failing tests pass. We drive it with [**mini-SWE-agent**](https://mini-swe-agent.com), pointed at the local `llama-server` router from page 6, and score with the official SWE-bench harness.

Everything runs as the `llm` service user. The reference model is `qwen3-coder-next`.

> **Architecture caveat (read first).** The DGX Spark / GX10 is **aarch64**. Both mini-SWE-agent and the SWE-bench harness hardcode `x86_64` Docker image names, which exit instantly with an *exec-format error* on this CPU. SWE-bench *does* publish arm64 images (`docker.io/swebench/sweb.eval.arm64.*`), so the fix is to make the tooling pick the host arch — see [§15.4](#154-the-aarch64-fixes). Without it nothing runs here.

## 15.1 What you need

- **Docker** running, and the `llm` user in the `docker` group:
  ```bash
  sudo usermod -aG docker llm     # log out/in (or new session) to take effect
  ```
- **Disk**: SWE-bench instance images are ~1–2 GB each; budget tens of GB for a real slice.
- The page-6 router up on `127.0.0.1:8080` with an API key (page 5).
- Versions this page was written against: mini-SWE-agent **2.4.0**, swebench **4.1.0**, Docker **29.2.1**.

## 15.2 Install (as `llm`)

```bash
sudo -iu llm python3 -m pipx install mini-swe-agent      # provides: mini, mini-extra
sudo -iu llm python3 -m pipx ensurepath                  # puts ~/.local/bin on PATH
```

## 15.3 Point it at the local model

mini-SWE-agent uses **LiteLLM**, which speaks any OpenAI-compatible endpoint. Put the model + endpoint in its global config so every run picks them up:

```bash
# /home/llm/.config/mini-swe-agent/.env  (chmod 600 — it holds the API key)
OPENAI_API_BASE=http://127.0.0.1:8080/v1
OPENAI_API_KEY=sk-...                 # a key from /etc/llama-server/api_keys.txt
MSWEA_MODEL_NAME=openai/qwen3-coder-next
MSWEA_COST_TRACKING=ignore_errors     # LiteLLM has no price for a local model; without this it aborts as fatal
```

The `openai/` prefix is what tells LiteLLM to treat `qwen3-coder-next` as a model served by the endpoint in `OPENAI_API_BASE`.

## 15.4 The aarch64 fixes

Three things must be true before a run completes here. Two are config (above: `docker` group, `MSWEA_COST_TRACKING`); the third is the **image architecture**, patched in two places:

| Package | File | Default | Fix |
|---|---|---|---|
| mini-SWE-agent (rollouts) | `minisweagent/run/benchmarks/swebench.py` | `sweb.eval.x86_64.<id>` | arch from `platform.machine()` |
| swebench (scoring) | `swebench/harness/run_evaluation.py` | `make_test_spec(... )` → `arch="x86_64"` | pass `arch=` from `platform.machine()` |

Both edits live **inside the venvs**, so they are lost on `pipx upgrade` / `pip install -U swebench`. The repo ships an **idempotent** re-patcher — run it after install and after any upgrade:

```bash
sudo -iu llm bash /path/to/repo/swebench/patch-arm64.sh
# [mini-swe-agent] patched (arm64-aware)
# [swebench harness] patched (arm64-aware)
```

It auto-locates both venvs (override with `MSWEA_VENV_PY` / `SB_VENV_PY`), only patches what's installed, and is safe to re-run (reports `already patched`).

## 15.5 Generate patches (rollouts)

```bash
sudo -iu llm bash -lc '
  mini-extra swebench \
    --subset lite --split test \
    --slice 0:1 \
    --workers 1 \
    --model openai/qwen3-coder-next \
    -o runs/smoke'
```

- `--subset` — `lite` (300), `verified` (500), `full`, … `--slice 0:N` takes the first N (always start small).
- `--workers N` — concurrency. Safe here: the router keeps **one** model resident and `llama-server` serves parallel slots, so workers share the loaded model instead of thrashing swaps.
- Output: `runs/smoke/preds.json`, mapping each instance id to the model's `model_patch`.

A rollout spins the instance's repo container, lets the model run shell commands until it submits a diff. Single-stream on `qwen3-coder-next` this is **~10–12 min/instance** (incl. the first-request model swap), so Lite/Verified end-to-end is hours-to-days — size with `--slice`.

## 15.6 Score the patches

Scoring is a **separate** step: apply each patch in the instance's container and run its tests. Use an isolated venv for the official harness (kept apart from the pipx agent venv):

```bash
sudo -iu llm bash -lc '
  python3 -m venv ~/sb-venv && ~/sb-venv/bin/pip install -q swebench
  ~/sb-venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-Bench_Lite \
    --predictions_path ~/runs/smoke/preds.json \
    --run_id myrun --max_workers 1 --cache_level instance'
```

Remember to run `patch-arm64.sh` again after creating `~/sb-venv` (it patches the harness too). The report is written to `openai__qwen3-coder-next.myrun.json`:

```json
{ "submitted_instances": 1, "completed_instances": 1,
  "resolved_instances": 1, "error_instances": 0,
  "resolved_ids": ["astropy__astropy-12907"] }
```

`resolved` = the patch made the failing tests pass (and kept the passing ones green). `error` (vs `unresolved`) means the harness itself failed — on this box that almost always means the arch patch isn't applied; re-run `patch-arm64.sh`.

**Cloud alternative — `sb-cli`.** If you don't want to maintain the local harness on arm64, generate `preds.json` locally and submit for scoring in SWE-bench's cloud:

```bash
sb-cli submit swe-bench_lite test --predictions_path preds.json --run_id myrun
```

This sidesteps the harness arch issue entirely (rollouts still run locally).

## 15.7 Verified end-to-end result

A single-instance smoke test passed the whole pipeline on this box:

| Stage | Result |
|---|---|
| Rollout (`astropy__astropy-12907`, arm64 container) | `Submitted` — non-empty patch, ~11.5 min |
| Patch quality | the canonical one-line fix in `astropy/modeling/separable.py` |
| Harness score (image reused) | **`resolved` 1/1**, ~49 s |

So `qwen3-coder-next` on the page-6 router can drive mini-SWE-agent against real SWE-bench instances and produce harness-verified fixes. This validates **function**, not a leaderboard number — run a sized `--slice` (then the full subset) for a real score.

## 15.8 The aarch64 reality: image coverage decides everything

The fix in §15.4 makes the tooling *ask* for arm64 images, but it can't conjure ones that don't exist. **Prebuilt arm64 image coverage is partial, and that — not model quality — is the binding constraint on this box.** Measured:

| Dataset | arm64 images | note |
|---|---|---|
| SWE-bench **Lite** (Python) | **52 / 300 (17 %)** | concentrated in django (27) + sympy (19); matplotlib/pytest/pylint/pydata/mwaskom/pallets have **zero** |
| SWE-bench **Multilingual** Java (druid, lucene, gson, javaparser, lombok, rxjava) | **0 / 43 (0 %)** | x86_64-only |

Instances without an arm64 image fail at `docker run` with **exit 125** (rollout never starts → empty patch) or a harness error. So **always pre-scan and filter** before a run:

```bash
# probe each instance id; keep the ones that have an arm64 image
docker manifest inspect docker.io/swebench/sweb.eval.arm64.<id-with __→_1776_>:latest
```

A practical consequence: a "5-model comparison on Lite slice 0:5" mostly measures *image availability* — 3 of those 5 instances have no arm64 image, so every model scores empty on them. Filter to the arm64-runnable set first, and remember a local Lite run is effectively a **django/sympy** benchmark.

### What we tried and rejected

- **qemu emulation** (run the x86_64 images under `binfmt`/`tonistiigi/binfmt --install amd64`). It *works* — full end-to-end Java rollout + score succeeded — but the rollout is **~30 min/instance** (the agent's shell commands over a large repo under emulation) vs ~10–12 min native. Too slow for anything but a one-off. **We removed it** (`tonistiigi/binfmt --uninstall qemu-x86_64`) so x86 images now fail fast instead of silently emulating.
- **Building arm64 images ourselves** (`prepare_images --namespace none --tag latest --env_image_tag latest`, with `make_test_spec` taught to honor host arch / `SWEBENCH_ARCH`). Viable in principle, but **for Java it does not yield a native image**: the SWE-bench `sweb.base.java.*` base is amd64-pinned, so the build produces an **amd64** image (slow, under emulation) that scoring rejects on an arm64-tag/amd64-content mismatch. For **Python** the arm64 base exists, so self-building *could* extend Lite past the 17 % prebuilt — that's the one place this lever pays off (not yet done here).

### Recommendation

- **Python, local:** run the arm64-runnable Lite subset natively. Optionally self-build more arm64 Python images to widen coverage.
- **Java / full coverage / leaderboard numbers:** don't do it on this box. Move **rollouts to x86** (Modal sandboxes via `swerex_modal`, with the model reached over the page-7 Cloudflare tunnel) and/or score with **`sb-cli`** in the cloud. The local box stays best for arm64 Python plus the bespoke page-12 Java harness.

---

[← Sampling & variance](14-sampling-and-variance.md) · [Index](README.md)
