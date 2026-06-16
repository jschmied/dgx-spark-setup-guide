# 13. Model evaluation: coding ability

[← Coding test](12-model-coding-test.md) · [Index](README.md) · [Next: Sampling & variance →](14-sampling-and-variance.md)

Results of the [page 12](12-model-coding-test.md) coding test — two tasks (Go concurrency, Spring stack), via the [`bench/`](bench/) harness. Each model runs at its **deployed** per-model sampling. The *how we got here* — the sampling experiments, the variance, the harness fixes — is on [page 14](14-sampling-and-variance.md); this page is just the latest results.

## 13.1 Setup

| | |
|---|---|
| Date | 2026-06-09 (llama.cpp b9571); `gemma-4-31B` added 2026-06-16 (b9641) |
| Hardware | GB10 (128 GB unified), router mode, one model resident |
| Sampling | each model at its **deployed** `models.ini` sampling (table below); **not** a single fixed temperature — see [page 14](14-sampling-and-variance.md) for why |
| Samples | **N=4** for the four fast local models. **Sonnet 4.6 †** is a single indicative sample (different harness, not re-run). |
| Verdict | **Neutral suite** = the model's *production* code passes an independent suite (the real signal). **Own** = the model's *delivered* tests also build & pass. |

Deployed sampling (`temp / top_p / top_k / repeat / min_p`; `top_p` off = nucleus disabled):

| Model | temp | top_p | top_k | repeat | min_p |
|---|---|---|---|---|---|
| `qwen3-coder-next`   | 0.7 | 0.8  | 20 | 1.05 | — |
| `qwen36-35b-a3b`     | 0.6 | 0.95 | 20 | —    | — |
| `gemma-4-26B-A4B`    | 1.0 | off  | 64 | —    | **0.1** |
| `gemma-4-31B`        | 1.0 | 0.95 | 64 | —    | — |

**Headline — two findings, one per task:**

1. **Spring/Java production logic was correct for every model on every sample** (neutral 4/4, 1/1 for the dense model). The only differentiator is whether each model's *own delivered tests* compile and pass.
2. **Go production correctness is sampling-sensitive — but tunable.** At the general-purpose recommended temps it is a coin-flip-or-worse, yet the right sampling fixes it: **min-p (top-p off)** took `gemma-4-26B-A4B` from 1/4 to **4/4** (at temp 1.0). For models not yet tuned, sample N times and keep the build-green one, or lower temp with a compile gate. The tuning story is on [page 14](14-sampling-and-variance.md).

## 13.2 Task A — Go cache: results

Neutral = production code passes the independent suite (idempotent `Close`, expired→miss, LRU order, `-race`). Own = delivered `cache.go`+`cache_test.go` compiles, `go vet`-clean, own tests pass.

| Model (sampling) | **Neutral (prod-correct)** | Own (delivered) | Notes |
|---|---|---|---|
| `gemma-4-26B-A4B` (min-p 0.1) | **4/4** ✅ | 0/4 | min-p fixed the variance (was 1/4 with top-p); own tests carry unused-symbol nits |
| `gemma-4-31B` (1.0, top-p 0.95) | **3/4** | 2/4 | reasoning model; the one neutral miss was a trivial unused-variable compile error (`k declared and not used`) |
| `qwen36-35b-a3b` (0.6) | 2/4 | 1/4 | best of the top-p models; documents `capacity==0`; fails with `undefined: K` (generics) |
| `qwen3-coder-next` (0.7) | 1/4 | 0/4 | `Close()` panics on 2nd call; non-compiling own test |
| `Sonnet 4.6 †` (default, N=1) | 1/1 | 16/17 | cleanest single sample; one self-inconsistent timing test |

## 13.3 Task B — Spring stack: results

Neutral = production logic passes an independent (entity-shape-agnostic) Mockito suite. Own = delivered `mvn test` is green.

| Model (sampling) | **Neutral (prod-correct)** | Own (delivered) | Notes |
|---|---|---|---|
| `gemma-4-26B-A4B` (min-p 0.1) | **4/4** ✅ | 3/4 | most reliable local Java deliverable, fewest tokens |
| `gemma-4-31B` (1.0, top-p 0.95) | 4/4 ✅ | 1/4 | production logic correct every sample; own tests over-specify `save()` (verify `times(1)` vs the legitimate 2 calls) |
| `qwen36-35b-a3b` (0.6) | 4/4 ✅ | 2/4 | when green: 7/7, explicit `save`, `@Version` asserted |
| `qwen3-coder-next` (0.7) | 4/4 ✅ | 0/4 | own test calls a `setId` the entity lacks → never compiles |
| `Sonnet 4.6 †` (default, N=1) | 1/1 ✅ | 12/12 | most thorough suite (version-on-insert, null-amount) |

## 13.4 Recommendation

- **Fastest capable local model → `gemma-4-26B-A4B`.** With **min-p 0.1** it is production-correct on both tasks (Go 4/4, Java 4/4), the most reliable Java deliverable, the fastest and lightest, and now MTP-accelerated (page 8 §8.10). The clear default for local coding.
- **Larger Gemma, reasoning variant → `gemma-4-31B`.** Same family as the 26B but a *reasoning* model (emits `reasoning_content` first, so give it generous `max_tokens`). Production-correct on Java (neutral 4/4) and Go **3/4** — the single Go miss a trivial unused-variable compile error — so just behind the 26B on Go, and its *own* delivered tests are frequently self-inconsistent (over-verify `save()`). Default to the faster, lighter 26B; reach for the 31B when you want the larger model's reasoning. Sampling: `temp 1.0 / top-p 0.95 / top-k 64`.
- **Best reasoning model → `qwen36-35b-a3b`.** Relative best of the top-p models on Go (2/4, documents `capacity==0`) and clean Spring builds; pays in reasoning tokens/latency.
- **High-throughput scaffolding → `qwen3-coder-next`** — cheapest, production logic usually right, but **gate every deliverable** (Go `Close` panic, Java never compiles its tests).
- **Frontier reference → `Sonnet 4.6 †`** when output quality outweighs keeping inference on-box.
- **Cross-cutting:** for deterministic code, don't trust a single sample at a general-purpose temperature. Use min-p where it helps (gemma), sample N times, or lower temp with a compile/test gate. The reasoning behind all of this is on [page 14](14-sampling-and-variance.md).

## 13.5 Adding a model

Add it to `models.ini`, set its sampling in `bench/lib.sh` (`sampling_for`), then:

```bash
cd bench
./run-samples.sh <new-id> 4 both     # N=4, both tasks → pass-rates
```

Append a row to each scorecard (§13.2, §13.3) with the **neutral pass-rate** as the headline, the **own** pass-rate, and any failure modes. If it is sampling-sensitive, see [page 14](14-sampling-and-variance.md) for the min-p / multi-sample playbook.

---

[← Coding test](12-model-coding-test.md) · [Index](README.md) · [Next: Sampling & variance →](14-sampling-and-variance.md)
