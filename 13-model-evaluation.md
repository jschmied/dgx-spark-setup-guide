# 13. Model evaluation: coding ability

[← Coding test](12-model-coding-test.md) · [Index](README.md)

Results of the [page 12](12-model-coding-test.md) coding test — two tasks (Go concurrency, Spring stack). Re-run that procedure to add a model.

## 13.1 Setup

| | |
|---|---|
| Date | 2026-06-08 |
| Hardware | GB10 (128 GB unified), router mode, one model resident |
| Sampling | `temperature 0.2`, `top_p 0.9` for all (the standard) — **except `gemma-4-26B-A4B ‡`**, run at its native `temp 1.0 / top_k 64 / top_p 0.95`; non-streaming (`max_tokens` 16000 for Task A, 20000 for Task B) |
| Task A | generic thread-safe TTL+LRU cache in Go — verdict `go vet` + `go test -race` (own + neutral + cross-run) |
| Task B | Spring Boot 3.3 + Hibernate/JPA + Mockito + JUnit 5 transfer module — verdict `mvn test` + neutral Mockito suite |
| Models | `qwen3-coder-next`, `qwen36-35b-a3b`, `ornstein36-27B ‡`, `ornstein36-35b-a3b`, `gemma-4-26B-A4B ‡` (local, served by the router); **Sonnet 4.6 †** added later |
| Samples | 1 per model per task (see caveats) |

> **† Sonnet 4.6 methodology caveat.** Sonnet was run as a **one-shot generator** (no tools, no iteration — `tool_uses: 0`) but through the **Claude Code agent harness** (its own system prompt) at **default sampling**, whereas the local models got a bare chat completion at `temperature 0.2`. The task, extraction, and objective verdict are identical, so treat the cross-vendor comparison as **indicative, not controlled**.
>
> **‡ Low-temperature loop caveat (Gemma 4 and the dense Ornstein).** Two models **loop at the standardized `temperature 0.2`** and deliver nothing on the reasoning-heavy Go task: `gemma-4-26B-A4B` (a known Gemma low-temp trait) and `ornstein36-27B` (which also looped at 0.6). Both have an **embedded `temp 1.0`** recommendation and deliver normally there, so Task A for these two was graded at their **native sampling** (`temp 1.0`, model-specific top-k/top-p). Their results are real but **off the common `temp 0.2` axis** — like Sonnet, treat the ranking as indicative. This was the trigger for the config audit that pinned each model's recommended sampling in the preset (page 8 §8.8).

**Headline:** across six models the *business logic* was almost always correct; what differed was whether the **delivered artifact** builds — and, for two models, whether they delivered at all at the standard temperature. `qwen36-35b-a3b` and `Sonnet 4.6` ship building, self-consistent projects on **both** tasks. Everyone else ships at least one artifact whose **own tests** are broken while the production logic is right (verified by the neutral suites): `qwen3-coder-next` (Go `Close` panic; Java won't compile), `ornstein36-35b-a3b` (Spring tests contradict its own service), `gemma-4-26B-A4B ‡` (one unused import in the Go test), and `ornstein36-27B ‡` (unused import + vars in the Go test). The last two **loop at `temp 0.2`** and were graded at their native `temp 1.0` — *sampling, not capability*: at the right temperature both deliver production-correct code. On speed: `gemma-4-26B-A4B` is the **fastest** model on the box (~74–85 t/s, no MTP), then `ornstein36-35b-a3b` (~75 with MTP); `qwen3-coder-next` is cheapest and `ornstein36-27B` by far slowest (~19 — see [page 8 §8.10–8.12](08-performance-tuning.md)).

## 13.2 Task A — Go cache: scorecard

| Check | qwen3-coder-next | qwen36-35b-a3b | ornstein36-27B | ornstein36-35b-a3b | gemma-4-26B-A4B ‡ | Sonnet 4.6 † |
|---|---|---|---|---|---|---|
| Compiles + `go vet` clean | ✅ | ✅ | ⚠️ vet flags unused import + vars | ✅ | ⚠️ vet flags unused `fmt` in test | ✅ |
| **Neutral suite** `go test -race` | ❌ **panics on test 1** | ✅ **all pass** | ✅ **all pass** | ✅ **all pass** | ✅ **all pass** | ✅ **all pass** |
| Own suite `go test -race` | ❌ FAIL (1 panic + 2 assertions) | ✅ PASS | ❌ **build fail** (unused `sync/atomic` + 2 vars) | ✅ **PASS** | ❌ **build fail** (unused `fmt` import) | ⚠️ **16/17** (1 timing test) |
| `Close()` idempotent (rule 5) | ❌ `close of closed channel` | ✅ closed-channel guard | ✅ **`sync.Once`** | ✅ **`sync.Once`** | ✅ **`sync.Once`** | ✅ **`sync.Once`** (+ concurrent-close test) |
| O(1) LRU | ✅ `container/list` | ✅ hand-rolled DLL | ✅ `container/list` | ✅ hand-rolled DLL | ✅ hand-rolled DLL | ✅ `container/list` |
| Get-expired no-promote (rule 3) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Lazy + janitor expiry | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ logic ok (see note) |
| `-race` clean concurrency | untestable (panic aborts binary) | ✅ | ✅ (via neutral) | ✅ | ✅ (via neutral) | ✅ |
| Assumptions + complexity deliverable | ❌ omitted | ✅ thorough, documented | ✅ thorough, documented | ✅ thorough, documented | ✅ thorough, documented | ✅ thorough, documented |
| Output | 4,011 tokens, no reasoning, fast | 14,315 tokens (mostly reasoning), 355 s | **loops @0.2 & @0.6**; @1.0 19,155 tok, ~17 min | 14,778 tokens (reasoning), 196 s | **loops @0.2**; @1.0 11,156 tok, 178 s | ~14,600 tokens |
| Test suite | larger (327 lines) but ≥2 **wrong** assertions | 159 lines, correct | correct logic, unused import + vars | 169 lines, correct | correct logic, 1 unused import | 17 tests, 1 self-inconsistent (timing) |

**Winner: `qwen36-35b-a3b` and `ornstein36-35b-a3b`** — both have a fully green own suite *and* neutral suite. `ornstein36-35b-a3b` matches `Sonnet 4.6`'s airtight `sync.Once` `Close` while also passing its own tests, and is the **fastest** to produce it (196 s at ~70 t/s). Three more are **production-correct** (neutral green) but ship a Go test that won't compile because of unused symbols: `gemma-4-26B-A4B ‡` (one unused import) and `ornstein36-27B ‡` (unused import + 2 vars) — both at their native `temp 1.0`, having looped at `temp 0.2` — plus `Sonnet 4.6` is red on one self-inconsistent timing test. `qwen3-coder-next` wins on speed/cost but its `Close` panics (§13.3). Every model except coder-next got the *production* logic right; the deliverable failures are all in the **tests**.

## 13.3 Task A: the decisive defect

`qwen3-coder-next` wrote `Close()` as `close(c.janitorStop)` with **no guard**, so a second call panics — directly violating rule 5 ("safe to call multiple times") and crashing the test binary before the concurrency test can even run.

`qwen36-35b-a3b` guarded it:

```go
func (c *Cache[K, V]) Close() {
    select {
    case <-c.stop:
        return        // already closed -> no-op
    default:
        close(c.stop)
    }
    c.ticker.Stop()
    c.wg.Wait()
}
```

(Minor note: this `select` guard is safe for *sequential* repeated calls as the spec requires, but two goroutines calling `Close` simultaneously could still race to `close`; a `sync.Once` would be airtight. Not exercised by the spec or the tests.)

## 13.4 Task A: fairness analysis

The first read ("coder fails its own tests") **overstated** the gap. Cross-running each model's tests against the other's implementation showed:

- **2 of coder-next's 3 self-failures were bugs in its *tests*, not its code.** Both independent implementations produce the value coder-next's assertions reject (e.g. its `TestLRUEviction` asserts `hits=3`; *both* models compute `4`). Coder-next's core get/set/LRU/expiry logic is sound in isolation.
- **`capacity==0` is ambiguous in the prompt.** coder-next chose "store nothing"; qwen36 chose "unlimited" — and qwen36 **documented** the choice in its assumptions. Neither is wrong; the neutral suite asserts nothing on it.

So the genuine, unambiguous implementation defect is the **`Close()` panic** — and that alone is enough to fail the neutral suite.

## 13.5 Task A: per-model notes

**qwen3-coder-next** (80B-A3B, Q4) — fast and concise; correct data structure (`container/list`), correct expiry/LRU/Get logic; answered directly with no reasoning pass. But shipped a real crash bug (`Close`), skipped the assumptions/complexity deliverable, and its own test suite carried incorrect assertions. Good default for throughput; review edge-case-heavy code before trusting it.

**qwen36-35b-a3b** (35B-A3B, Q8, reasoning) — reasoned through the edge cases (idempotent close, lazy-delete on expiry) and **documented** the ambiguous `capacity==0` decision; hand-rolled a correct O(1) doubly-linked-list LRU; the only one to deliver the assumptions + complexity write-up; passes both its own and the neutral suite race-clean. Cost ~6× the tokens and 355 s, almost all in the thinking pass — that reasoning is *what bought the correctness*.

**ornstein36-27B ‡** (27B dense, Q6, reasoning) — **a sampling story, not a capability one.** At the standardized `temperature 0.2` it fell into a verbatim repetition loop (`**One Edge Case:** Set key A (update)… Correct.` ×N) and emitted no code; a re-run at `temp 0.6` looped too. The fix was its **embedded recommended `temp 1.0`** (found in the later config audit): at `temp 1.0 / top_k 20 / top_p 0.95` it delivered a 19,155-token answer (~17 min — it is the slowest model) whose **production code is correct** — neutral suite race-clean, airtight `sync.Once` `Close`, `container/list` O(1) LRU, a documented `capacity<=0` choice, and a full assumptions + complexity write-up. Like Gemma, its **own test won't compile**: an unused `sync/atomic` import plus two unused vars (`h`, `m`). So it is *not* the "can't finish" model the first pass suggested — it joins the broken-delivered-test club, gated behind a temperature that must be ≈1.0. The earlier `temp 0.2`/`0.6` loops are why its preset now pins `temp = 1.0` (page 8 §8.8/§8.10).

**ornstein36-35b-a3b** (35B-A3B MoE, Q8, reasoning) — the cleanest Go result of the local models: it terminated its reasoning cleanly even at `temp 0.2` (no loop, unlike its dense sibling), passed `go vet`, its **own** suite, and the **neutral** suite all race-clean. It used the airtight `sync.Once` for `Close` (matching Sonnet), a hand-rolled O(1) doubly-linked list, **documented** its `capacity<=0 = store nothing` choice, and delivered a thorough assumptions + complexity write-up (it even noted the janitor's O(N) scan and the NTP-clock-drift caveat). Fastest of the reasoning models to produce it (196 s, ~70 t/s with MTP). A co-winner on Task A — and notably more robust to low temperature than the dense merge from the same family.

**gemma-4-26B-A4B ‡** (Gemma 4, MoE, Q4 QAT, different vendor) — **two-sided result.** At the standardized `temperature 0.2` it fell into a repetition loop (the same 4-line block ×164) and delivered nothing — the same low-temp failure mode as the dense Ornstein, and equally **recoverable**: re-run at Gemma's native `temp 1.0` and it produces a genuinely good answer. That answer is production-correct (neutral suite race-clean), uses the airtight `sync.Once` `Close`, a hand-rolled O(1) doubly-linked list, **documents** its `capacity<=0 = store nothing` choice, and includes a full assumptions + complexity write-up. The **only** blemish is that its `cache_test.go` imports `fmt` without using it, so `go vet` flags it and the own suite won't compile — a one-token fix, but as delivered it's red. The mildest "broken delivered test" on the board, and the only model whose Go failure is a pure style nit rather than a logic or lifecycle bug. Also the fastest to generate (178 s at ~75–85 t/s).

**Sonnet 4.6 †** — the cleanest `Close` of the models *that delivered* (`sync.Once`, plus it wrote its own concurrent-close test); correct `container/list` LRU, correct expiry/LRU/stats; passes the neutral suite race-clean and documents `capacity==0 = unbounded`. Its **one own-suite failure is self-inflicted**: it clamps the janitor interval to a 1 s minimum, but `TestJanitorRemovesExpired` only waits 200 ms — so its own test and its own impl disagree on timing (the janitor logic itself is fine; entries also expire lazily). Net: production-correct, but the delivered `go test` is red on that one timing test — a milder version of the same "self-inconsistent test" failure mode coder-next showed.

## 13.6 Task B — Spring stack: scorecard

| Check | qwen3-coder-next | qwen36-35b-a3b | ornstein36-27B | ornstein36-35b-a3b | gemma-4-26B-A4B ‡ | Sonnet 4.6 † |
|---|---|---|---|---|---|---|
| Deliverable `mvn test` | ❌ **does not compile** (test sources) | ✅ **BUILD SUCCESS — 7/7** | ✅ **BUILD SUCCESS — 7/7** | ❌ **FAIL — own tests 2/7** (compiles) | ✅ **BUILD SUCCESS — 7/7** | ✅ **BUILD SUCCESS — 12/12** |
| — Mockito unit tests | — (compile fail) | ✅ 5/5 | ✅ 5/5 | ⚠️ 3/5 (2 self-inconsistent) | ✅ 5/5 | ✅ 8/8 |
| — `@DataJpaTest` (real Hibernate+H2) | — | ✅ 2/2 (`@Version` increment) | ✅ 2/2 (`@Version` increment) | ✅ 2/2 (`@Version` increment) | ✅ 2/2 (`@Version` increment) | ✅ 4/4 (`@Version` increment **+ =0 on insert**) |
| **Neutral** Mockito suite vs *production* code | ✅ **5/5** | ✅ **5/5** | ✅ **5/5** (acc() adapted to 3-arg ctor) | ✅ **5/5** | ✅ **5/5** | ✅ **5/5** |
| Self-consistent (entity API matches its own tests) | ❌ test calls `Account.setId()` it never defined | ✅ `setId` + `getVersion` present | ✅ own tests compile & pass; `Account(id,owner,balance)` ctor | ❌ tests assume short-circuit lookup; service looks up both eagerly | ✅ own tests compile & pass | ✅ sets id via reflection (clean entity) |
| Mockito strict-stub discipline | n/a (didn't compile) | ✅ documented + correct | ✅ documented + correct | ⚠️ a failing test trips `verifyNoMoreInteractions` on the 2nd lookup | ✅ correct | ✅ correct (AssertJ + ArgumentCaptor) |
| Spec adherence | Spring Boot 3.3.2, skipped file labels | Spring Boot 3.3.5, all files + assumptions | Spring Boot 3.3.0, all files + assumptions | Spring Boot 3.3.5, all files + assumptions | Spring Boot 3.3.0, all files (AssertJ) | Spring Boot 3.3.4, all files + assumptions |
| Output | 3,034 tokens, no reasoning, 86 s | 8,408 tokens (reasoning), 199 s | 10,013 tokens (reasoning), 429 s | 9,759 tokens (reasoning), 126 s | 4,767 tokens, 69 s (fastest) | ~13,500 tokens |

**Winner: `Sonnet 4.6` / `qwen36-35b-a3b`** — both build and pass; Sonnet's suite is the most thorough (12 tests, incl. version-on-insert, null-amount, both unknown-account directions). **`ornstein36-27B` and `gemma-4-26B-A4B ‡` also build green (7/7)** — Gemma did it in the fewest tokens and fastest wall-clock of any model. `ornstein36-35b-a3b` and `qwen3-coder-next` both fail `mvn test` despite correct production logic (neutral 5/5) — coder-next doesn't compile; ornstein-35B compiles but ships unit tests that contradict its own service.

## 13.7 Task B: what happened

Both models' **business logic is correct** — the identical neutral Mockito suite passes **5/5 against each model's production code** (transfer, insufficient-funds, unknown-account, non-positive, same-account).

The difference is the *delivered artifact*:

- **qwen3-coder-next** wrote sound production code (idiomatic dirty-checking via `@Transactional`, `BigDecimal.compareTo`, proper custom exceptions) but its **own test sources don't compile**, for two self-inconsistencies: `assertThrows` used with no `import static org.junit.jupiter.api.Assertions.*`, and `Account` has no `setId(...)` although its test calls it. Since the task's acceptance criterion is "tests pass with `mvn test`", it fails as delivered.
- **qwen36-35b-a3b** built and passed everything, kept entity and tests consistent, used **explicit `save()`** (so persistence is verifiable with `verify(repo).save(...)`), exercised real Hibernate via `@DataJpaTest` including the `@Version` increment, and **explicitly anticipated the traps** in its assumptions (BigDecimal+`compareTo`, `@Version` as `Integer`, *"stub only methods actually invoked"* for strict stubs).

- **Sonnet 4.6 †** also built green (12/12) and was the **most thorough** — it kept the entity clean (no `setId`) and set the id via **reflection** in the unit test so the test still compiles, added a `@Version`-is-0-on-insert assertion and a null-amount case, and used AssertJ + `ArgumentCaptor`. Neutral suite 5/5.

- **ornstein36-27B** — here it **delivered even at the standard `temp 0.2`** (the Spring task is less reasoning-heavy than the Go one, so it didn't trip the loop), and well: `mvn test` BUILD SUCCESS 7/7 (5 Mockito + 2 `@DataJpaTest`), real Hibernate on H2 with the `@Version` increment asserted across a flush/round-trip, `BigDecimal.compareTo` throughout, explicit `save(from)`/`save(to)`, strict-stub-clean unit tests, and a genuinely good assumptions write-up (it explicitly called out strict stubs, `@Version`, and BigDecimal scale). Its only deviation: an `Account(Long id, String owner, BigDecimal balance)` constructor instead of the 2-arg one the neutral suite assumes — a valid choice the prompt didn't pin down, so the neutral `acc()` helper was adapted by one line (passing `null` id), exactly the kind of entity-shape accommodation Sonnet's reflection trick handles. Neutral suite then 5/5. Slowest to produce it (429 s at ~19 t/s), but correct.

- **ornstein36-35b-a3b** — production logic correct (neutral 5/5) but its **own unit tests fail (2/7), so `mvn test` is red.** It compiles cleanly (unlike coder-next); the failure is a self-inconsistency. Its `TransferService` looks up *both* accounts up front (`findById(fromId)` **and** `findById(toId)`) before any check — a fine, arguably better design — but its tests assume short-circuit/lazy lookup: `testInsufficientFunds` stubs only account 1, so the eager second lookup returns empty and throws `AccountNotFoundException` before the balance check (the test expected `InsufficientFundsException`); `testUnknownAccount`'s `verifyNoMoreInteractions` trips on that same second lookup. Same family failure mode as Task A's dense sibling and as coder-next: the *delivered tests* don't match the model's own code. (It still got `compareTo`, explicit `save`, `@Version`-increment assertion, and a clean `@DataJpaTest` right.)

- **gemma-4-26B-A4B ‡** — clean **BUILD SUCCESS 7/7** at its native sampling, and the most efficient of all: 4,767 tokens in 69 s. Correct production logic (neutral 5/5), `BigDecimal.compareTo`, explicit `save`, `@Version` increment asserted via `@DataJpaTest`, AssertJ throughout, 2-arg constructor matching the neutral suite. No self-inconsistencies here — a notable contrast to its Go task, where a stray unused import sank its own suite. (As with all Gemma runs, this was at `temp 1.0`; at 0.2 it would have looped.)

Shape across the field on Task B: coder-next is fast with correct core logic but a non-compiling test artifact; **ornstein-35B compiles with correct logic but self-contradicting tests**; qwen36, Sonnet **and gemma** reason through it and ship green builds (gemma in the fewest tokens by far). The two Ornstein merges still diverge per-task — the dense 27B aced Task B (7/7) while shipping a non-compiling Go test, and the MoE 35B aced Task A while shipping broken Spring tests — but neither is a non-delivery once each is run at the temperature its card recommends.

## 13.8 Recommendation (across both tasks)

- **Correctness-critical / edge-case-heavy delivery, local/offline** → `qwen36-35b-a3b`. It shipped building, passing, self-consistent code on both tasks. Accept the latency and token cost.
- **Best raw quality if a frontier API is acceptable** → `Sonnet 4.6 †` produced the most thorough Java suite and the most robust Go `Close`; its only blemish was one self-inconsistent Go timing test.
- **High-throughput / interactive coding, scaffolding, refactors** → `qwen3-coder-next` for speed — but **add a compile/test pass**: on both tasks its production logic was right while its delivered tests were broken.
- **`ornstein36-35b-a3b` → promising but verify every deliverable.** It is the **fastest model on the box** (~70–75 t/s with MTP, page 8 §8.11) and produced the best Go result of any model (own + neutral green, `sync.Once` Close). But on the Spring task it shipped unit tests that contradict its own service, so `mvn test` was red despite correct logic. Strong candidate for high-quality local generation **with a mandatory compile/test gate** — the same discipline coder-next needs.
- **`ornstein36-27B ‡` → only at `temp 1.0`, and only if you accept the latency.** Run at its embedded `temp 1.0` it delivers production-correct code on both tasks (Spring 7/7; Go neutral-green with a non-compiling test). But it is **by far the slowest** model on the box (dense, ~19 t/s even with MTP) and the most sampling-fragile — at `temp 0.2`/`0.6` it loops and emits nothing, so it is risky for unattended use unless the sampling is pinned. Of the two Ornstein merges, prefer the **MoE 35B**: same family, far faster, robust at low temp. Keep the dense 27B only when you specifically want its style and can give it `temp 1.0` plus an output/compile check.
- **`gemma-4-26B-A4B ‡` → fast, capable, but pin the sampling.** The **fastest and lightest** model here (~74–85 t/s, ~14 GB, no MTP), production-correct on both tasks (both neutral suites green) — full green Spring build, and on Go only a one-token unused-import nit. The catch is operational: it **must run at `temp ≈ 1.0`**; the standardized `temp 0.2` sent it into a repetition loop. Excellent default for fast local generation *provided* its preset sampling is set (page 8 §8.12) — and, like the others, gate the output with a compile/test pass.

This mirrors the throughput data on [page 8](08-performance-tuning.md): `gemma-4-26B-A4B` leads on speed (no MTP needed), then `ornstein36-35b-a3b`; qwen36 is faster per token but spends far more tokens (reasoning); coder-next is the cheaper path when its answer is correct — and a quick `go test`/`mvn test` catches the cases where it isn't; the dense ornstein is the slowest and least predictable. Reserve `Sonnet 4.6` for when output quality matters more than keeping inference on-box.

## 13.9 Caveats

- **Single sample per model per task** — indicative, not statistical. A firm ranking needs 3–5 samples each (page 12 §12.9). One sample is one data point; the value here is the *kind* of failure each model shows, not a precise ordering.
- **Sampling is not one-size-fits-all — and getting it wrong looks like incompetence.** The single biggest lesson of this evaluation: at the fixed benchmark `temp 0.2`, both `gemma-4-26B-A4B` and `ornstein36-27B` fell into repetition loops and emitted *nothing* — and the first write-up wrongly concluded the dense Ornstein "can't finish." Re-run at each model's **embedded recommended `temp 1.0`** and both deliver production-correct code. A fixed benchmark temperature is fair for *comparison* but can completely misrepresent a model whose architecture expects a different operating point. Always check the model card / GGUF `general.sampling` before concluding a model can't do a task — that audit is exactly what now pins each model's sampling in the preset (page 8 §8.8).
- Two tasks already reduce single-task bias. The two Ornstein merges still diverge per-task (dense aces Spring + non-compiling Go test; MoE aces Go + broken Spring tests), but — once sampled correctly — neither is a non-delivery, so the divergence is about *which test breaks*, not *whether it answers*.
- Cross-task consistency is the strongest signal here: the *same* failure mode — **correct production logic, broken delivered tests** — showed up for coder-next (Go + Java), ornstein-35B (Java), ornstein-27B and gemma (unused-symbol Go tests), and, more mildly, Sonnet (Go timing). Only `qwen36-35b-a3b` shipped green on both. The recurring lesson: **the production code was almost always right; the tests were where models broke.**
- **† Sonnet 4.6 and ‡ the two low-temp loopers are not on the common axis** (see §13.1): Sonnet ran through a different harness at default sampling; `gemma-4-26B-A4B` and `ornstein36-27B` were graded at their native `temp 1.0` because `temp 0.2` loops. All produced real, gradeable artifacts under an identical task and verdict, but treat their rankings relative to the `temp 0.2` models as indicative, not controlled.

## 13.10 Adding a model

Run [page 12](12-model-coding-test.md) for both tasks with `MODEL=<new-id>` (after adding it to `models.ini`), then append a column to each scorecard:

| Task A — Go | <new-model> | | Task B — Spring | <new-model> |
|---|---|---|---|---|
| Neutral `go test -race` | _pass/fail_ | | Deliverable `mvn test` | _pass/fail_ |
| `Close()` idempotent | _✅/❌_ | | Neutral Mockito suite | _pass/fail_ |
| O(1) LRU | _✅/❌_ | | Self-consistent build | _✅/❌_ |
| Deliverables complete | _✅/❌_ | | `@Version` tested | _✅/❌_ |
| Tokens / wall time | _… / …_ | | Tokens / wall time | _… / …_ |

---

[← Coding test](12-model-coding-test.md) · [Index](README.md)
