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

---

[← Sampling & variance](14-sampling-and-variance.md) · [Index](README.md)
