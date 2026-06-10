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

`ornstein36-35b-a3b` was the most volatile model on Go (1/4 at top-p, a different bug each sample), so it got a full sweep — and unlike Gemma, min-p alone wasn't enough; **temperature was the other half.**

First pass, min-p at its recommended `temp 0.6` and at `temp 1.0`:

| `ornstein36-35b-a3b` (N=4) | Go neutral | Java neutral |
|---|---|---|
| top-p 0.95 (temp 0.6, original) | 1/4 | 4/4 |
| min-p 0.1, top-p off (temp 0.6) | 3/4 | 2/4 |
| min-p 0.05, top-k 40 (temp 1.0) | 2/4 | 3/4 |

min-p helped Go but with a Go↔Java seesaw and no clear winner — and min-p **0.05** was distinctly worse on Java than **0.1** (a 10-shot Java sweep confirmed it: min-p 0.1 passed 6/6 across temps 0.2–1.0, min-p 0.05 only 1/3, the looser tail letting a bug through). So min-p was pinned at 0.1; the remaining variable was temperature.

A **Go temperature sweep** (top-k 20, min-p 0.1, N=3 each) showed a clean trend — *lower is better*:

| temp (top-k 20, min-p 0.1) | Go neutral |
|---|---|
| 0.3 | 3/3 |
| 0.4 | 2/3 |
| 0.5 | 1/3 |
| 0.6 | 2/3 |

Confirming the two best at **N=4 on both tasks**:

| config | Go neutral | Java neutral |
|---|---|---|
| temp 0.2, top-k 20, min-p 0.1 | **4/4** | 2/4 |
| **temp 0.3, top-k 20, min-p 0.1** | **4/4** | **3/4** |

**Sweet spot: `temp 0.3 / top-k 20 / min-p 0.1` (top-p off)** — Go **1/4 → 4/4**, Java 3/4. That is the model's deployed config now.

**The one residual Java failure is not a sampling problem.** Across *every* config — even temp 0.2 — roughly one Java sample in three or four ships an `InsufficientFundsException` that uses `BigDecimal` without importing it (`cannot find symbol`). It is a **sampling-resistant model defect**, a one-line missing import that no temperature or truncation reliably removes — and that a `mvn test` catches in seconds. This is the cleanest illustration of the page's closing rule: **tune sampling for the bugs sampling causes, and gate the build for the bugs it doesn't.**

**A counter-example where min-p does *not* generalize: `qwopus36-35b-a3b`.** A *separate* Qwen3.6-35B-A3B finetune (Jackrong's "Qwopus", Q4_K_M) — the **same `qwen35moe` arch** as `ornstein36-35b-a3b`, so the natural assumption was that the sibling's `temp 0.3 / min-p 0.1` sweet spot would transfer. It does not. Four configs, N=4 on both tasks:

| `qwopus36-35b-a3b` (N=4) | Go neutral | Java neutral | Failure character |
|---|---|---|---|
| **temp 0.6 / top-p 0.95** (Qwen3.6 default) | **1/4** | **4/4** | real attempts: 1 non-delivery, a `syntax error: unexpected &&`, an `e.Value` capitalization bug |
| temp 0.3 / min-p 0.1, top-p off (sibling's sweet spot) | 1/4 | 4/4 | **3/4 looped** to the 26k-token cap emitting *no code* (own tests 2/4 → 0/4) |
| temp 0.6 / min-p 0.1, top-p off | 0/4 | 3/4 | worse on both — min-p didn't tame the tail, and one Java sample regressed |
| temp 1.0 / top-p 0.95 (loop-prone-model hypothesis) | 1/4 | 3/4 | **no loops** (all delivered), but a different Go compile bug each fail, and one Java drops to the missing-`import` defect |

So for Qwopus the plain Qwen3.6 default (`temp 0.6 / top-p 0.95`) is the **best of the four** — the *only* config that holds Java at 4/4 while tying for the best Go. The sweep cleanly separates two distinct failure regimes: lowering temperature to 0.3 pushes it into the *reasoning-loop / non-delivery* trap (the Gemma/dense-Ornstein behaviour of §14.1, **not** the Ornstein-35b cure), while raising it to 1.0 *removes* the loops (every sample delivered) but still doesn't lift Go and costs a Java sample. Adding min-p at 0.6 helps nothing and nicks Java. **Go stays a coin-flip-at-best (1/4) under every config tried** — a model whose Go weakness is **not** a sampling artifact you can tune away; gate it on `go build`. The lesson sharpens §14.5's: **same base model and same architecture do not predict the operating point** — Qwopus and ornstein-35b are both `qwen35moe` Qwen3.6 finetunes yet want opposite temperatures. Sweep each model from scratch.

**Takeaway:** min-p is a real lever, but model-specific and often paired with the right temperature — and sometimes it helps nothing. Gemma needed only min-p (at its required `temp 1.0`); `ornstein36-35b-a3b` needed min-p **and** a low temperature (0.3); its same-arch cousin `qwopus36-35b-a3b` wanted **neither** (low temp made it loop) and kept the plain default. Sweep per model — and always **gate every deliverable on `go build` / `mvn test`.**

## 14.6 The harness and the playbook

Everything here is reproducible from [`bench/`](bench/):

- **Multi-sample runner** — `./run-samples.sh <model> <N> both` reports pass-rates, not a single ✅/❌.
- **Content-based Go extraction** — `cache.go` = the block with `func New`/`type Cache`; `cache_test.go` = the block with `func Test`. Robust to models that emit extra fenced blocks.
- **Entity-shape-agnostic neutral Java suite** — reflection-based `acc()` (§14.3).
- **Single-shot sampling sweep** — `./sweep-java.sh <model>` runs one Java shot per config across a temp/top-k/min-p grid (how §14.5 was mapped).
- **Experiment overrides** — `BENCH_FORCE_TEMP`, `BENCH_FORCE_TOPP` (set `1.0` to disable nucleus), `BENCH_FORCE_TOPK`, `BENCH_FORCE_MINP`, e.g.
  ```bash
  BENCH_FORCE_TEMP=0.3 BENCH_FORCE_TOPP=1.0 BENCH_FORCE_TOPK=20 BENCH_FORCE_MINP=0.1 ./run-samples.sh <model> 4 both
  ```

**Playbook when a model "can't code":**
1. Check it isn't *looping* — a low temperature (or a low-temp-sensitive model) emits no code. Raise temperature / use the card's value.
2. If it loops at the recommended temperature, that model needs `temp ≈ 1.0` (Gemma, dense Ornstein).
3. If it delivers but with a *different bug each run*, it's tail-sampling noise — try **min-p 0.1 with top-p off**, and if that isn't enough, **sweep temperature** (lower is usually better for deterministic code: `ornstein36-35b-a3b` needed both — min-p 0.1 *and* temp 0.3). Use min-p 0.1, not 0.05 — 0.05's looser tail still leaks bugs.
4. Always grade on multiple samples and gate the deliverable on `go build` / `mvn test`. The production logic is usually right; the delivered tests are where models break.

---

[← Model evaluation](13-model-evaluation.md) · [Index](README.md)
