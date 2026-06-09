# 13. Model evaluation: coding ability

[← Coding test](12-model-coding-test.md) · [Index](README.md)

Results of the [page 12](12-model-coding-test.md) coding test — two tasks (Go concurrency, Spring stack), run through the [`bench/`](bench/) harness. Re-run that procedure to add a model.

## 13.1 Setup

| | |
|---|---|
| Date | 2026-06-09 (re-run at per-model recommended sampling; supersedes the 2026-06-08 fixed-`temp 0.2` run) |
| Hardware | GB10 (128 GB unified), router mode, one model resident; llama.cpp b9571 |
| Sampling | **each model at its own recommended sampling** (the value pinned in `models.ini` / model card / GGUF `general.sampling`), *not* a single fixed temperature — see the table below and `bench/lib.sh` |
| Samples | **N=3 per task** for the four fast models, plus the N=1 full-sweep run (so up to **4 samples**); **N=1** for the dense `ornstein36-27B` (≈27 min/sample, too slow to multi-sample). **Sonnet 4.6 †** is the prior single indicative sample (not re-run). |
| Task A | generic thread-safe TTL+LRU cache in Go — verdict `go vet` + `go test -race` (own suite + **neutral** suite) |
| Task B | Spring Boot 3.3 + Hibernate/JPA + Mockito + JUnit 5 transfer module — verdict `mvn test` (own) + **neutral** Mockito suite (now entity-shape-agnostic, §13.4) |

Per-model sampling used (`temp / top_p / top_k / repeat`):

| Model | temp | top_p | top_k | repeat |
|---|---|---|---|---|
| `qwen3-coder-next`   | 0.7 | 0.8  | 20 | 1.05 |
| `qwen36-35b-a3b`     | 0.6 | 0.95 | 20 | —    |
| `ornstein36-27B`     | 1.0 | 0.95 | 20 | —    |
| `ornstein36-35b-a3b` | 0.6 | 0.95 | 20 | —    |
| `gemma-4-26B-A4B`    | 1.0 | 0.95 | 64 | —    |

> **Why per-model sampling.** An earlier version of this page ran every model at a fixed `temperature 0.2`. That **misrepresented** models whose architecture expects a higher operating point — Gemma 4 and the dense Ornstein simply *loop* at low temp and delivered nothing, which the first write-up wrongly read as "can't code." Running each model at the sampling it was tuned for removes that artifact. The trade-off, made explicit below, is that the recommended temps are *general-purpose* (0.6–1.0), and a code task is low-entropy — so single-sample variance is high. That is why this run uses multiple samples and reports **pass-rates**, not a single ✅/❌.

> **† Sonnet 4.6 is an indicative reference, not a controlled data point.** It was run once, through the Claude Code agent harness (its own system prompt) at default sampling, as a one-shot generator (`tool_uses: 0`). Same task and objective verdict, but a different harness and sampling — treat its ranking as indicative. It was **not** re-run for this revision; its results are carried over.

**Headline — two findings, one per task:**

1. **Spring/Java production logic was correct for *every* model on *every* sample.** The neutral Mockito suite passes 4/4 (1/1 for the dense model) across the board. What differs is only whether each model's **own delivered tests** compile and pass — `qwen3-coder-next` never ships a compiling test (calls a `setId` its entity lacks), while gemma/qwen36/ornstein-35b ship a green build most of the time.
2. **Go production correctness is a coin-flip-or-worse at these sampling settings.** The headline neutral suite passes at best **2/4** (`qwen36-35b-a3b`) and **1/4** for everyone else — and every failure is a *different* real bug (a missing generic type parameter, a non-existent `atomic.Uint64.Inc`, a `c.stopChan` typo, a `Close()` panic, a generics syntax error). The earlier impression of solid Go competence came from near-greedy `temp 0.2` single samples; at the recommended temps the variance is real. **Lower the temperature (with a compile gate) for deterministic code generation** — see §13.4.

On speed (see [page 8 §8.9–8.12](08-performance-tuning.md)): `gemma-4-26B-A4B` is the fastest (~74–96 t/s, now with MTP via a separate assistant draft), then `ornstein36-35b-a3b` (~75 with MTP); `qwen3-coder-next` is the cheapest in tokens (no reasoning pass); the dense `ornstein36-27B` is by far the slowest (~19 t/s).

## 13.2 Task A — Go cache: results

Pass-rates over all samples (N=4 for fast models = the N=3 multi-sample run + the full-sweep sample; N=1 for the dense Ornstein and Sonnet). **Neutral** = the model's *production* code passes the independent suite (idempotent `Close`, expired→miss, LRU order, `-race`). **Own** = the model's *delivered* `cache.go`+`cache_test.go` compiles, `go vet`-clean, and its own tests pass.

| Model (sampling) | **Neutral (prod-correct)** | Own (delivered) | Failure modes seen | Structure when it passed |
|---|---|---|---|---|
| `qwen3-coder-next` (0.7) | **1/4** | 0/4 | `Close()` panic (`close of closed channel`); `key declared and not used` | `container/list` O(1) LRU |
| `qwen36-35b-a3b` (0.6) | **2/4** | 1/4 | `undefined: K` — `entry[V any]` missing the `K` type parameter (×2) | hand-rolled DLL; documents `capacity==0` |
| `ornstein36-27B` (1.0, N=1) | **1/1** | 0/1 | own test: unused `sync/atomic` import + 2 unused vars | `sync.Once` `Close`, `container/list` |
| `ornstein36-35b-a3b` (0.6) | **1/4** | 1/4 | `atomic.Uint64.Inc` (no such method); runtime panic; `syntax error: missing type constraint` | `sync.Once`, hand-rolled DLL |
| `gemma-4-26B-A4B` (1.0) | **1/4** | 1/4 | `element.Value` called as a function; `c.stopChan` undefined (×2) | `sync.Once`, hand-rolled DLL |
| `Sonnet 4.6 †` (default, N=1) | **1/1** | 16/17 | one self-inconsistent timing test (janitor clamps to 1 s, test waits 200 ms) | `sync.Once` + own concurrent-close test |

**Takeaway:** no local model is reliable on this task at its recommended sampling. `qwen36-35b-a3b` is the relative best (2/4) and the only one that, when it passes, also documents the ambiguous `capacity==0`. The two reasoning Ornstein merges and gemma each produced a *different* compile/runtime error on most samples — the model "knows" the algorithm (it gets the structure and the `sync.Once` `Close` right when it compiles) but trips on Go generics/standard-library details under sampling noise.

## 13.3 Task A: variance is the result

The single most important thing this task shows is **sampling variance on a low-entropy problem**. Three of five local models swung between a clean pass and a hard compile failure across just four samples, and each failure was distinct:

- `gemma-4-26B-A4B`: clean once, then `element.Value(...)` (treating `container/list.Element.Value` — an `any` field — as a function), then a `c.stopChan` field that the struct never declares.
- `ornstein36-35b-a3b`: `c.evictions.Inc()` (Go's `atomic.Uint64` has `.Add`, not `.Inc`), a runtime panic, and a generics syntax error — three different bugs in three samples.
- `qwen36-35b-a3b`: `type entry[V any]` that references `K` in its `key K` field without declaring `[K comparable, ...]`.

These are not capability ceilings — the same models produce correct code on other samples and on the easier Java task. They are what a general-purpose temperature (0.6–1.0) does to a task whose correct answer is essentially unique. **For production code generation, sample N times and keep the build-green one, or drop the temperature and gate on `go build`/`go test`.**

## 13.4 Task A & B: making the neutral suites fair

Two harness lessons, both now fixed in [`bench/`](bench/):

- **Go extraction is content-based.** `cache_test.go` is the block containing `func Test`; `cache.go` is the block with `func New`/`type Cache`. This is robust to models that emit extra fenced blocks (usage examples, an API sketch) — a positional "block 1 / block 2" rule mis-splits those.
- **The neutral Java suite must not assume an entity shape.** The first version hardcoded `new Account(owner, balance)`. Models legitimately chose other shapes — `Account(Long id, String owner, BigDecimal balance)` (qwen36, the Ornsteins) or no-arg + setters — and the suite then *failed to compile*, which looked like a logic failure but was a **measurement artifact**. The neutral `acc()` now constructs `Account` by **reflection** (tries `(String,BigDecimal)`, `(Long,String,BigDecimal)`, then no-arg + setters), mirroring the reflection trick Sonnet used in its own test. With that fix, **every model's Spring production logic passes the neutral suite** — the earlier qwen36 "0/3" and ornstein-35b "2/3" Java failures were entirely this artifact, not defects.

## 13.5 Task A: per-model notes

**qwen3-coder-next** (80B-A3B, Q4) — fast and token-cheap (no reasoning pass; ~3.5–16 k tokens). Gets the data structure right (`container/list`) but is the least reliable on the unambiguous requirement: its `Close()` either panics on a second call (`close of closed channel`, violating rule 5) or its delivered test won't build, so it passed the neutral suite only once in four and its own suite never. Good for throughput; **gate every deliverable on a compile/test pass.**

**qwen36-35b-a3b** (35B-A3B, Q8, reasoning) — the relative winner (2/4 neutral) and the only model that, on a passing sample, hand-rolls a correct O(1) DLL *and* documents the ambiguous `capacity==0`. Its characteristic failure is a generics slip (`entry[V any]` missing `K`). Spends the most tokens (reasoning) for that correctness.

**ornstein36-27B ‡** (27B dense, Q6, reasoning) — runs only at `temp 1.0` (it loops at 0.6/0.2 — why its preset pins 1.0, page 8 §8.8). At N=1 its production code passed the neutral suite (race-clean, `sync.Once` `Close`, `container/list`, documented `capacity<=0`), but its delivered test won't compile (unused `sync/atomic` import + 2 vars). By far the slowest model (~27 min for this answer), so it was not multi-sampled.

**ornstein36-35b-a3b** (35B-A3B MoE, Q8, reasoning) — the most volatile on Go: a different compile/runtime bug on each of its samples (`atomic.Uint64.Inc`, a panic, a generics syntax error), passing the neutral suite only once in four. When it compiles it uses `sync.Once` and a hand-rolled DLL. A clean reminder that its prior "co-winner" status was a single low-temp sample.

**gemma-4-26B-A4B** (Gemma 4, MoE, Q4 QAT, different vendor) — fastest and cheapest (~5–14 k tokens), but on Go it passed the neutral suite only once in four, with two distinct bugs otherwise (`element.Value` misuse, `c.stopChan` typo). It must run at `temp ≈ 1.0` (loops lower), which is part of why its Go output is noisy. Strong on Java (below), weak/variable on Go.

**Sonnet 4.6 †** — the cleanest single sample of any model: `sync.Once` `Close` plus its own concurrent-close test, correct `container/list` LRU and expiry, neutral suite green, documents `capacity==0`. Its one own-suite failure (16/17) is a self-inconsistent timing test (clamps the janitor to 1 s but waits 200 ms). Indicative only (different harness/sampling).

## 13.6 Task B — Spring stack: results

| Model (sampling) | **Neutral (prod-correct)** | Own (delivered `mvn test`) | Notes |
|---|---|---|---|
| `qwen3-coder-next` (0.7) | **4/4 ✅** | 0/4 | own test calls `Account.setId(...)` the entity never defines, and/or omits the JUnit 5 static import → never compiles |
| `qwen36-35b-a3b` (0.6) | **4/4 ✅** | 2/4 | when green: 7/7, explicit `save`, `@Version` increment asserted; documents strict-stub discipline |
| `ornstein36-27B` (1.0, N=1) | **1/1 ✅** | 1/1 | BUILD SUCCESS 7/7; real Hibernate `@Version` round-trip; strict-stub-clean |
| `ornstein36-35b-a3b` (0.6) | **4/4 ✅** | 2/4 | failures are self-inconsistent unit tests (eager double-lookup vs tests assuming short-circuit) |
| `gemma-4-26B-A4B` (1.0) | **4/4 ✅** | 3/4 | most reliable Java deliverable of the local models; one sample used a wrong `@DataJpaTest` import (`...data.jpa` vs `...orm.jpa`) |
| `Sonnet 4.6 †` (default, N=1) | **1/1 ✅** | 12/12 | most thorough suite (version-on-insert, null-amount, both unknown-account directions); reflection-set id |

**Takeaway:** on the framework task, **production logic is universally correct** (neutral green everywhere). The differentiator is the *delivered artifact*: `qwen3-coder-next` never ships a compiling test, the two mid reasoning models ship a green build about half the time, and `gemma-4-26B-A4B` is the most reliable local Java deliverable (3/4) and the fastest/cheapest to produce it.

## 13.7 Task B: what happened

The business rules — positive-amount, distinct accounts, both-exist, sufficient-funds, debit/credit with `BigDecimal.compareTo` — were implemented correctly by every model (neutral 5/5 against each one's production code). The delivered-build failures are all in the *tests* or the *entity surface*:

- **qwen3-coder-next** — sound production code (dirty-checking via `@Transactional`, custom exceptions, `BigDecimal`), but its **own test sources never compile**: an `assertThrows` with no `import static org.junit.jupiter.api.Assertions.*`, and a `setId(...)` the entity doesn't expose. 0/4 delivered.
- **qwen36-35b-a3b** — when green (2/4), an exemplary build: explicit `save()`, `@DataJpaTest` exercising the real `@Version` increment, and assumptions that call out strict stubs and `BigDecimal` scale. Its red samples are own-test inconsistencies, not logic.
- **ornstein36-35b-a3b** — production correct (neutral 4/4) but ships unit tests that contradict its own service: the service looks up *both* accounts up front, while `testInsufficientFunds`/`testUnknownAccount` assume short-circuit lookup, so a stubbed-only-account-1 test throws `AccountNotFoundException` first and `verifyNoMoreInteractions` trips on the eager second lookup. 2/4 delivered.
- **gemma-4-26B-A4B** — the most reliable local Java deliverable (3/4), in the fewest tokens; correct `compareTo`, explicit `save`, `@Version` assertion, AssertJ. Its one red sample imported `@DataJpaTest` from the wrong package.
- **ornstein36-27B** (N=1) — BUILD SUCCESS 7/7 even on a single sample; the Spring task is less reasoning-heavy than the Go one, so it didn't trip the loop that plagues it on Go.
- **Sonnet 4.6 †** — the most thorough (12/12): clean entity (id set by reflection in the test), version-on-insert and null-amount cases, AssertJ + `ArgumentCaptor`.

## 13.8 Recommendation (across both tasks)

- **Best overall local model → `qwen36-35b-a3b`.** Relative winner on Go (2/4, and the only one that documents `capacity==0`) and a clean Spring build when green, with universally-correct production logic on both tasks. Accept the reasoning-pass latency and token cost.
- **Fastest capable local model → `gemma-4-26B-A4B`.** Fastest and cheapest, the most reliable *Java* deliverable (3/4), production-correct on both tasks. The catch: its **Go** output is noisy (1/4) and it must run at `temp ≈ 1.0`. Excellent default for framework/boilerplate work; gate Go output hard.
- **High-throughput scaffolding → `qwen3-coder-next`** — but it is the least reliable on deliverables (Go `Close` panic, Java never compiles its tests). Only with a **mandatory compile/test gate**; its production logic is usually right even when its tests aren't.
- **Of the two Ornstein merges, prefer the MoE `ornstein36-35b-a3b`** — same family, far faster than the dense 27B, robust at low temp — but its Go output is volatile (1/4, a different bug each sample); verify every deliverable. Keep the dense **`ornstein36-27B`** only when you want its style and can afford `temp 1.0` plus the ~27-min latency.
- **Frontier reference → `Sonnet 4.6 †`** when output quality matters more than keeping inference on-box: the cleanest single Go sample and the most thorough Java suite.
- **Cross-cutting: for deterministic code generation, do not rely on a single sample at the card's general-purpose temperature.** Either sample N times and keep the build-green one, or lower the temperature and gate on `go build`/`mvn test`. This run's whole point is that sampling, not raw capability, dominated the delivered results.

This mirrors the throughput data on [page 8](08-performance-tuning.md): `gemma-4-26B-A4B` leads on speed (now with MTP), then `ornstein36-35b-a3b`; `qwen36-35b-a3b` is slower per token but spends them on the reasoning that buys its edge; `qwen3-coder-next` is the cheapest when correct; the dense Ornstein is slowest and least predictable.

## 13.9 Caveats

- **Variance dominates on Go.** Even N=4 is small; the spread (a different bug per sample) is the signal. A firm ranking needs more samples per cell (page 12 §12.9). Treat the Go pass-rates as "unreliable at recommended temps," not as precise orderings.
- **Recommended ≠ optimal-for-code.** The per-model temps here are the card/`general.sampling` *general-purpose* values (chosen partly to avoid the low-temp loop in Gemma and the dense Ornstein). A pure code benchmark would arguably use the lowest non-looping temperature per model; that would raise correctness but is a different experiment. This page deliberately uses the *deployed* sampling so it predicts what the router actually returns.
- **Neutral suites test production logic, deliberately decoupled from delivered tests.** The recurring pattern — correct production code, broken or non-compiling *delivered* tests — held on both tasks. The neutral Java suite is now entity-shape-agnostic (§13.4) so it no longer penalizes valid entity designs.
- **`ornstein36-27B` is N=1** (too slow to multi-sample) and **`Sonnet 4.6 †` is a single indicative sample** through a different harness at default sampling. Treat both as data points, not distributions.

## 13.10 Adding a model

Add it to `models.ini`, set its recommended sampling in `bench/lib.sh` (`sampling_for`), then:

```bash
cd bench
./run-samples.sh <new-id> 3 both     # N=3, both tasks → pass-rates
# or a single pass of each task:
./run-go.sh <new-id> ; ./run-java.sh <new-id>
```

Then append a row to each scorecard (§13.2, §13.6) with the **neutral pass-rate** (production correctness) as the headline, the **own** pass-rate (delivered build), the failure modes seen, and token cost.

---

[← Coding test](12-model-coding-test.md) · [Index](README.md)
