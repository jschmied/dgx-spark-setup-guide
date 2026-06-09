# 14. Sampling & variance: how the eval was tuned

[← Model evaluation](13-model-evaluation.md) · [Index](README.md)

[Page 13](13-model-evaluation.md) reports the *results*. This page is the messy part behind them: the sampling mistakes, the run-to-run variance, the harness bugs, and the experiments that led to each model's deployed sampling. If you only want the scorecard, stay on page 13. Read this when a model "can't code" — because the cause is usually **sampling, not capability**.

## 14.1 The first mistake: one fixed temperature for everyone

The first version of the evaluation ran every model at a single fixed `temperature 0.2`. That looked rigorous — identical sampling, "about capability not luck." It was wrong.

At `temp 0.2`, **`gemma-4-26B-A4B` and the dense `ornstein36-27B` fall into verbatim repetition loops** and emit *no code at all*. The first write-up read that as "the dense Ornstein can't finish the Go task." It can — at the temperature its model card recommends (`temp ≈ 1.0`) it delivers production-correct code. A fixed low temperature is not neutral: it silently breaks any model whose architecture expects a higher operating point.

**Fix:** run each model at its own recommended sampling (`models.ini` / model card / GGUF `general.sampling`), encoded in `bench/lib.sh`. That removed the loop artifact — and immediately exposed the *next* problem.

## 14.2 Recommended temps trade loops for variance

The recommended temps are *general-purpose* (0.6–1.0). A coding task is low-entropy — the correct answer is essentially unique — so a higher temperature introduces real run-to-run variance. Running **N=3** samples per model (plus the single full-sweep sample → N=4) made this unmistakable: on the Go task, three of five models swung between a clean pass and a hard compile failure, and **each failure was a different real bug**:

| Model (Go, top-p) | Neutral pass-rate | Distinct failures seen across samples |
|---|---|---|
| `qwen36-35b-a3b` (0.6) | 2/4 | `undefined: K` — `entry[V any]` missing the `K` type parameter (×2) |
| `qwen3-coder-next` (0.7) | 1/4 | `Close()` panic (`close of closed channel`); `key declared and not used` |
| `ornstein36-35b-a3b` (0.6) | 1/4 | `atomic.Uint64.Inc` (no such method); a runtime panic; `syntax error: missing type constraint` |
| `gemma-4-26B-A4B` (1.0) | 1/4 | `element.Value` called as a function; `c.stopChan` undefined (×2) |

These are not capability ceilings — the same models produce correct code on other samples and on the easier Java task. They are what a general-purpose temperature does to a problem whose correct answer is unique. **Lesson: never grade a deterministic code task on a single sample at a general-purpose temperature.** Java, by contrast, was stable: production logic correct on every model, every sample.

## 14.3 A harness bug that looked like a model failure

While grading Task B, `qwen36-35b-a3b` scored **0/3** on the neutral Java suite — alarming for a model that ships clean Spring builds. The cause was **my suite, not the model**: the neutral `NeutralTransferTest` hardcoded `new Account(owner, balance)`, but qwen36 (and the Ornsteins) legitimately defined `Account(Long id, String owner, BigDecimal balance)`. The suite *failed to compile*, which looked like a logic failure but was a measurement artifact.

**Fix:** the neutral `acc()` helper now builds `Account` by **reflection** — it tries `(String,BigDecimal)`, then `(Long,String,BigDecimal)`, then no-arg + setters — mirroring the trick Sonnet used in its own test. With that fix, **every model's Spring production logic passes** (neutral 4/4 across the board). The earlier qwen36 "0/3" and ornstein-35b "2/3" Java failures vanished entirely. The general lesson: a neutral suite must test *logic*, not assume a specific surface the model was never told to expose.

## 14.4 Tuning Gemma's Go reliability: top-p vs temp vs min-p

Gemma was the most interesting case: production-correct on Java every time, but only **1/4** on Go at its recommended `temp 1.0 / top-p 0.95`. Three sampling regimes, N=4 each, both tasks:

| Sampling (temp 1.0 unless noted) | Go neutral | Java neutral | Loops? |
|---|---|---|---|
| top-p 0.95 (original recommended) | 1/4 | 4/4 | none |
| `temp 0.5` + top-p 0.95 | mixed — s1 looped (26k tok, no code), s2 clean | mixed — s1 clean, s2 looped (24k tok) | **yes, both tasks** |
| **min-p 0.1, top-p off** | **4/4** ✅ | **4/4** ✅ | none |

- **Lowering temperature is the wrong lever.** `temp 0.5` pushes Gemma toward its low-temp loop on the longer-reasoning Go task (and intermittently on Java) — total non-delivery, worse than a buggy answer.
- **min-p is the right lever.** `min_p 0.1` keeps only tokens with probability ≥ 10 % of the top token's, which adaptively truncates the long tail that produced the fatal rare tokens (`c.stopChan`, `element.Value`-as-function) — *without* over-constraining into loops the way a blanket low temperature does. Go production-correctness went **1/4 → 4/4**; Java stayed perfect; reasoning stayed on (~30k chars/sample); no loops.

That is why Gemma's deployed sampling is **`temp 1.0 / top-k 64 / min-p 0.1`, top-p disabled** (page 8 §8.12).

## 14.5 Does min-p generalise? (other models)

min-p clearly fixed Gemma. To see whether it is a general cure for the recommended-temp Go variance, the same change (top-p off, `min-p 0.1`, each model's own temperature) was run N=4 on the next-most-volatile model:

`ornstein36-35b-a3b` was the most volatile model on Go (1/4 at top-p, a different bug each sample). Run again with **min-p 0.1, top-p off** at its own `temp 0.6`:

| `ornstein36-35b-a3b` (N=4) | Go neutral | Java neutral |
|---|---|---|
| top-p 0.95 (temp 0.6) | 1/4 | 4/4 |
| **min-p 0.1, top-p off** (temp 0.6) | **3/4** | **2/4** |

min-p **helped Go** (1/4 → 3/4 — the remaining failure was a generics `missing type constraint`) but **hurt Java** (4/4 → 2/4): two samples shipped an `InsufficientFundsException` that uses `BigDecimal` without importing it (`cannot find symbol`). So min-p is **not a universal cure** — it transformed Gemma (Go 1/4 → 4/4, Java untouched at 4/4) but only partly helped this model and traded away Java reliability. Two caveats: N=4 is small, and this ran at `temp 0.6` (Gemma's win was at `temp 1.0`), so temperature and min-p interact.

**Takeaway:** min-p is a real lever for tail-sampling variance, but its effect is **model-specific — test it per model, don't blanket-apply.** Gemma adopted it (a clear win on its required `temp 1.0`); `ornstein36-35b-a3b` keeps its top-p config (its Java was cleaner there, and it isn't the recommended pick anyway — page 13). The general rule stands regardless of sampling: **gate every deliverable on `go build` / `mvn test`.**

## 14.6 The harness and the playbook

Everything here is reproducible from [`bench/`](bench/):

- **Multi-sample runner** — `./run-samples.sh <model> <N> both` reports pass-rates, not a single ✅/❌.
- **Content-based Go extraction** — `cache.go` = the block with `func New`/`type Cache`; `cache_test.go` = the block with `func Test`. Robust to models that emit extra fenced blocks.
- **Entity-shape-agnostic neutral Java suite** — reflection-based `acc()` (§14.3).
- **Experiment overrides** — `BENCH_FORCE_TEMP`, `BENCH_FORCE_TOPP` (set `1.0` to disable nucleus), `BENCH_FORCE_MINP`, e.g.
  ```bash
  BENCH_FORCE_TOPP=1.0 BENCH_FORCE_MINP=0.1 ./run-samples.sh <model> 4 both
  ```

**Playbook when a model "can't code":**
1. Check it isn't *looping* — a low temperature (or a low-temp-sensitive model) emits no code. Raise temperature / use the card's value.
2. If it loops at the recommended temperature, that model needs `temp ≈ 1.0` (Gemma, dense Ornstein).
3. If it delivers but with a *different bug each run*, it's tail-sampling noise — try **min-p 0.1 with top-p off** before concluding anything.
4. Always grade on multiple samples and gate the deliverable on `go build` / `mvn test`. The production logic is usually right; the delivered tests are where models break.

---

[← Model evaluation](13-model-evaluation.md) · [Index](README.md)
