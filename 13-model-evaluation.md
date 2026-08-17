# 13. Model evaluation: coding ability

[← Coding test](12-model-coding-test.md) · [Index](README.md) · [Next: Sampling & variance →](14-sampling-and-variance.md)

Results of the [page 12](12-model-coding-test.md) coding test — two tasks (Go concurrency, Spring stack), via the [`bench/`](bench/) harness. Each model runs at its **deployed** per-model sampling. The *how we got here* — the sampling experiments, the variance, the harness fixes — is on [page 14](14-sampling-and-variance.md); this page is just the latest results.

## 13.1 Setup

| | |
|---|---|
| Date | 2026-06-09 (llama.cpp b9571); `gemma-4-31B` added 2026-06-16 (b9641); **`qwen36-27b` rows corrected + `step-37` added 2026-07-17** |
| Hardware | GB10 (128 GB unified), router mode, one model resident |
| Sampling | each model at its **deployed** `models.ini` sampling (table below); **not** a single fixed temperature — see [page 14](14-sampling-and-variance.md) for why |
| Samples | **N=4** for the five local models (the dense `qwen36-27b` is the slow one, ~12–25k tokens/sample). **`step-37`** is **N=1** (20 min/response — see §13.2). **Sonnet 4.6 †** is a single indicative sample (different harness, not re-run). |
| Verdict | **Neutral suite** = the model's *production* code passes an independent suite (the real signal). **Own** = the model's *delivered* tests also build & pass. |

> ### ⚠️ 2026-07-17 — the `qwen36-27b` rows were harness artifacts, now corrected
> Both of the dense 27B's rows understated it, for **two different** harness reasons — neither of them the model:
> - **Java (§13.3) 3/4 → 4/4 neutral, 2/4 → 3/4 own.** Sample 3 labelled each file *inside* its code fence
>   (`src/main/java/.../Account.java` as body line 1), which `extract_java.py` wrote verbatim → every file died at
>   `[1,1]` and the production logic was **never compiled**. `extract_java.py` now strips a bare path label.
>   Same samples, re-scored — no model re-run. The old note blaming "the sample whose Go run also overran" was
>   wrong: `java_s3` finished `fin=stop` with 11,380 chars.
> - **Go (§13.2) 1/4 → 2/4 neutral.** The old sample 3 was `fin=length`, `tok=26000`, **`content=0`** — a token-cap
>   non-delivery scored as a model FAIL. Re-run **N=4 at a 60k cap** (`BENCH_MAXTOK_GO`), all four delivered
>   `fin=stop`. Fresh samples, so 1/4→2/4 is *partly* sampling noise; the durable point is that the new figure
>   contains **zero non-deliveries**. Raw: `bench/results-27b-recap-2026-07-17/` (originals kept in `bench/results/`).
>
> **No other row moves.** The extractor fix was diffed old-vs-new over **all 33 stored Java samples**: exactly one
> (`java_qwen36-27b_s3`) changes; every other sample extracts byte-identically. Models that label a file in its
> *own* fence (`qwen3-coder-next` does, 9×) were never broken — the extractor already skipped a label-only block;
> only a label *inside* the content fence breaks the build. The other four models were re-scored from their stored
> samples and reproduce their published rows.
>
> Cf. the [SWE-bench Docker-cache confound](appendix/nvfp4-agentic-coding.md) — same lesson: **check `finish_reason`
> and whether the code compiled at all before crediting a model with a failure.**

Deployed sampling (`temp / top_p / top_k / repeat / min_p`; `top_p` off = nucleus disabled):

| Model | temp | top_p | top_k | repeat | min_p |
|---|---|---|---|---|---|
| `qwen3-coder-next`   | 0.7 | 0.8  | 20 | 1.05 | — |
| `qwen36-35b-a3b`     | 0.6 | 0.95 | 20 | —    | — |
| `gemma-4-26B-A4B`    | 1.0 | off  | 64 | —    | **0.1** |
| `gemma-4-31B`        | 1.0 | 0.95 | 64 | —    | — |
| `qwen36-27b`         | 1.0 | 0.95 | 20 | —    | — |
| `step-37`            | 1.0 | 0.95 | 64 | —    | — |

`step-37` additionally **requires** `chat_template_kwargs.reasoning_effort` (`low`\|`medium`\|`high`; benched at
**`medium`** via `BENCH_REASONING_EFFORT`). Its template always opens `<think>` and there is no off-switch —
**unconditioned it spends the whole budget reasoning and returns empty content** (see §13.2).

**Headline — two findings, one per task:**

1. **Spring/Java production logic was correct for every local model on every sample** — neutral **4/4 across the board**, now genuinely without exception (the dense `qwen36-27b`'s 3/4 was an extraction artifact, corrected 2026-07-17). The main differentiator is whether each model's *own delivered tests* compile and pass.
2. **Go production correctness is sampling-sensitive — but tunable.** At the general-purpose recommended temps it is a coin-flip-or-worse, yet the right sampling fixes it: **min-p (top-p off)** took `gemma-4-26B-A4B` from 1/4 to **4/4** (at temp 1.0). For models not yet tuned, sample N times and keep the build-green one, or lower temp with a compile gate. The tuning story is on [page 14](14-sampling-and-variance.md).

## 13.2 Task A — Go cache: results

Neutral = production code passes the independent suite (idempotent `Close`, expired→miss, LRU order, `-race`). Own = delivered `cache.go`+`cache_test.go` compiles, `go vet`-clean, own tests pass.

| Model (sampling) | **Neutral (prod-correct)** | Own (delivered) | Notes |
|---|---|---|---|
| `gemma-4-26B-A4B` (min-p 0.1) | **4/4** ✅ | 0/4 | min-p fixed the variance (was 1/4 with top-p); own tests carry unused-symbol nits |
| `gemma-4-31B` (1.0, top-p 0.95) | **3/4** | 2/4 | reasoning model; the one neutral miss was a trivial unused-variable compile error (`k declared and not used`) |
| `qwen36-35b-a3b` (0.6) | 2/4 | 1/4 | best of the top-p models; documents `capacity==0`; fails with `undefined: K` (generics) |
| `qwen3-coder-next` (0.7) | 1/4 | 0/4 | `Close()` panics on 2nd call; non-compiling own test |
| `qwen36-27b` (1.0, top-p 0.95) | **2/4** | 1/4 | **corrected 2026-07-17** (was 1/4 — the old sample 3 was a 26k-cap non-delivery, not a model failure). Re-run at a 60k cap, all 4 delivered `fin=stop`, 12–25k tokens. The two misses are genuine compile errors: `key declared and not used`, `item … is not a type` |
| `step-37` (1.0, top-p 0.95, effort=medium, **N=1**) | 0/1 | 0/1 | used `atomic.Uint64` but omitted `sync/atomic` from its imports → doesn't build. **33.8k tokens ≈ 20 min for the one response.** N=1 on a task that is a coin-flip for everyone — does not separate it from the fleet |
| `Sonnet 4.6 †` (default, N=1) | 1/1 | 16/17 | cleanest single sample; one self-inconsistent timing test |

> **Reasoning models need a token cap that fits their thinking, or they score as failures they didn't earn.**
> `step-37` unconditioned: 26000/26000 tokens, `finish=length`, **`content_len=0`** — the harness then compiled an
> empty file and reported `vet=FAIL own=FAIL neutral=FAIL`. Its reasoning was coherent and on-task the whole way;
> it simply never reached the answer channel. Set `reasoning_effort` (§13.1) and raise `BENCH_MAXTOK_GO` — at
> `medium` it still spent **33.8k**, well past the harness's 26k default. The same cap cost `qwen36-27b` a Go
> sample. **Always check `fin=` before reading a verdict.**

## 13.3 Task B — Spring stack: results

Neutral = production logic passes an independent (entity-shape-agnostic) Mockito suite. Own = delivered `mvn test` is green.

| Model (sampling) | **Neutral (prod-correct)** | Own (delivered) | Notes |
|---|---|---|---|
| `gemma-4-26B-A4B` (min-p 0.1) | **4/4** ✅ | 3/4 | most reliable local Java deliverable, fewest tokens |
| `gemma-4-31B` (1.0, top-p 0.95) | 4/4 ✅ | 1/4 | production logic correct every sample; own tests over-specify `save()` (verify `times(1)` vs the legitimate 2 calls) |
| `qwen36-35b-a3b` (0.6) | 4/4 ✅ | 2/4 | when green: 7/7, explicit `save`, `@Version` asserted |
| `qwen3-coder-next` (0.7) | 4/4 ✅ | 0/4 | own test calls a `setId` the entity lacks → never compiles |
| `qwen36-27b` (1.0, top-p 0.95) | **4/4** ✅ | **3/4** | **corrected 2026-07-17** (was 3/4 neutral, 2/4 own). Sample 3 labelled files *inside* the fence → `extract_java.py` wrote the path as body line 1 → nothing compiled. Same samples re-scored with the fixed extractor: production logic correct on **all 4** |
| `step-37` (1.0, top-p 0.95, effort=medium, **N=1**) | 1/1 ✅ | 1/1 | full 9-file Maven project; own 5/5 + JPA 2/2 + neutral Mockito 5/5. Its best showing — but N=1, and 10.5k tokens ≈ 5.6 min |
| `Sonnet 4.6 †` (default, N=1) | 1/1 ✅ | 12/12 | most thorough suite (version-on-insert, null-amount) |

## 13.4 Recommendation

- **Fastest capable local model → `gemma-4-26B-A4B`.** With **min-p 0.1** it is production-correct on both tasks (Go 4/4, Java 4/4), the most reliable Java deliverable, the fastest and lightest, and now MTP-accelerated (page 8 §8.10). The clear default for local coding.
- **Larger Gemma, reasoning variant → `gemma-4-31B`.** Same family as the 26B but a *reasoning* model (emits `reasoning_content` first, so give it generous `max_tokens`). Production-correct on Java (neutral 4/4) and Go **3/4** — the single Go miss a trivial unused-variable compile error — so just behind the 26B on Go, and its *own* delivered tests are frequently self-inconsistent (over-verify `save()`). Default to the faster, lighter 26B; reach for the 31B when you want the larger model's reasoning. Sampling: `temp 1.0 / top-p 0.95 / top-k 64`.
- **Best reasoning model → `qwen36-35b-a3b`.** Relative best of the top-p models on Go (2/4, documents `capacity==0`) and clean Spring builds; pays in reasoning tokens/latency.
- **High-throughput scaffolding → `qwen3-coder-next`** — cheapest, production logic usually right, but **gate every deliverable** (Go `Close` panic, Java never compiles its tests).
- **Dense base Qwen3.6-27B → `qwen36-27b`.** **Re-assessed 2026-07-17 — it is not the weakest; the old rows were harness artifacts.** Java **4/4** neutral (production logic correct on every sample, matching the best local models) and Go **2/4**, level with `qwen36-35b-a3b` and above `qwen3-coder-next`. Still the slowest to *generate* (dense; ~34–40 t/s with DFlash, page 8 §8.8) and its *own* delivered tests remain weak (Go own 1/4) — so gate every deliverable. Its Go misses are ordinary compile errors (`key declared and not used`), the same failure mode as the rest of the fleet.
- **Step-3.7-Flash → `step-37`: not recommended, on latency.** 288x7.4B MoE at Q3_K_XL (~89 GB — evicts prod, ~6 min to load, `-np 1`, won't batch). At `reasoning_effort=medium` it spent **33.8k tokens ≈ 20 min on a single Go response**. Java was clean (neutral + own, N=1); Go missed on a forgotten `sync/atomic` import. Quality is **unsettled** (N=1 on a coin-flip task), but the wall-clock already rules it out for agentic work: ~30 calls/instance puts SWE-bench in the *days*. Revisit only at `reasoning_effort=low`, and only after an N=4 minibench.
- **Frontier reference → `Sonnet 4.6 †`** when output quality outweighs keeping inference on-box.
- **Cross-cutting:** for deterministic code, don't trust a single sample at a general-purpose temperature. Use min-p where it helps (gemma), sample N times, or lower temp with a compile/test gate. The reasoning behind all of this is on [page 14](14-sampling-and-variance.md).

## 13.5 Adding a model

Add it to `models.ini`, set its sampling in `bench/lib.sh` (`sampling_for`), then:

```bash
cd bench
./run-samples.sh <new-id> 4 both     # N=4, both tasks → pass-rates
```

**If it is a reasoning model, do this first** — otherwise you will score the harness, not the model:

```bash
# 1. Raise the caps. The defaults (Go 26k / Java 24k) are NOT enough: step-37 spent 33.8k
#    at effort=medium, and the 26k cap already cost qwen36-27b a Go sample.
export BENCH_MAXTOK_GO=60000 BENCH_MAXTOK_JAVA=60000
# 2. If its template exposes a thinking budget, set it (step-37: low|medium|high).
export BENCH_REASONING_EFFORT=medium
./run-samples.sh <new-id> 4 both
```

Then **read `fin=` in the output before believing any verdict**. `fin=length` with empty content is a
non-delivery — the harness compiles an empty file and reports `FAIL` for a model that never answered.
Re-run it with a bigger cap; do not score it.

Append a row to each scorecard (§13.2, §13.3) with the **neutral pass-rate** as the headline, the **own** pass-rate, and any failure modes. If it is sampling-sensitive, see [page 14](14-sampling-and-variance.md) for the min-p / multi-sample playbook.

---

[← Coding test](12-model-coding-test.md) · [Index](README.md) · [Next: Sampling & variance →](14-sampling-and-variance.md)
