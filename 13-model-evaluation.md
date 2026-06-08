# 13. Model evaluation: coding ability

[← Coding test](12-model-coding-test.md) · [Index](README.md)

Results of the [page 12](12-model-coding-test.md) coding test — two tasks (Go concurrency, Spring stack). Re-run that procedure to add a model.

## 13.1 Setup

| | |
|---|---|
| Date | 2026-06-08 |
| Hardware | GB10 (128 GB unified), router mode, one model resident |
| Sampling (identical for all) | `temperature 0.2`, `top_p 0.9`, non-streaming (`max_tokens` 16000 for Task A, 20000 for Task B) |
| Task A | generic thread-safe TTL+LRU cache in Go — verdict `go vet` + `go test -race` (own + neutral + cross-run) |
| Task B | Spring Boot 3.3 + Hibernate/JPA + Mockito + JUnit 5 transfer module — verdict `mvn test` + neutral Mockito suite |
| Models | `qwen3-coder-next`, `qwen36-35b-a3b`, `ornstein36-27B` (local, served by the router); **Sonnet 4.6 †** added later |
| Samples | 1 per model per task (see caveats) |

> **† Sonnet 4.6 methodology caveat.** Sonnet was run as a **one-shot generator** (no tools, no iteration — `tool_uses: 0`) but through the **Claude Code agent harness** (its own system prompt) at **default sampling**, whereas the local models got a bare chat completion at `temperature 0.2`. The task, extraction, and objective verdict are identical, so treat the cross-vendor comparison as **indicative, not controlled**.

**Headline:** of the four, three get the *business logic* right and differ only on whether the **delivered artifact** builds and passes; the fourth (`ornstein36-27B`) **never delivered Task A at all**. `qwen36-35b-a3b` and `Sonnet 4.6` ship building, self-consistent projects on both tasks; `qwen3-coder-next` has sound production logic but ships a broken artifact in both (Go: `Close` panic; Java: test sources don't compile); `ornstein36-27B` ships a clean Spring project (Task B) but on the Go task its reasoning pass **runs away and never emits code** (a degenerate repetition/CoT loop at both standardized and recommended sampling). `qwen3-coder-next` is far the cheapest/fastest. `ornstein36-27B` is also by far the **slowest** (dense model, ~19 t/s even with MTP — see [page 8 §8.10](08-performance-tuning.md)).

## 13.2 Task A — Go cache: scorecard

| Check | qwen3-coder-next | qwen36-35b-a3b | ornstein36-27B | Sonnet 4.6 † |
|---|---|---|---|---|
| Compiles + `go vet` clean | ✅ | ✅ | ❌ **no code emitted** | ✅ |
| **Neutral suite** `go test -race` | ❌ **panics on test 1** | ✅ **all pass** | ❌ **no deliverable** | ✅ **all pass** |
| Own suite `go test -race` | ❌ FAIL (1 panic + 2 assertions) | ✅ PASS | ❌ no deliverable | ⚠️ **16/17** (1 timing test) |
| `Close()` idempotent (rule 5) | ❌ `close of closed channel` | ✅ closed-channel guard | — (never reached) | ✅ **`sync.Once`** (+ concurrent-close test) |
| O(1) LRU | ✅ `container/list` | ✅ hand-rolled DLL | — | ✅ `container/list` |
| Get-expired no-promote (rule 3) | ✅ | ✅ | — | ✅ |
| Lazy + janitor expiry | ✅ | ✅ | — | ✅ logic ok (see note) |
| `-race` clean concurrency | untestable (panic aborts binary) | ✅ | — | ✅ |
| Assumptions + complexity deliverable | ❌ omitted | ✅ thorough, documented | ❌ never finished | ✅ thorough, documented |
| Output | 4,011 tokens, no reasoning, fast | 14,315 tokens (mostly reasoning), 355 s | **16,000 tok @0.2 / 24,000 @0.6 — 100 % reasoning, 0 content**; 12.5 / 20.5 min | ~14,600 tokens |
| Test suite | larger (327 lines) but ≥2 **wrong** assertions | 159 lines, correct | none (looped before emitting) | 17 tests, 1 self-inconsistent (timing) |

**Winner: `qwen36-35b-a3b`** (only one whose own suite is fully green). `Sonnet 4.6` is logically correct with the most robust `Close`, but its delivered `go test` is red on one self-inconsistent timing test (§13.5). `qwen3-coder-next` wins on speed/cost. **`ornstein36-27B` does not finish** — its reasoning pass never terminates into a code answer (§13.5).

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

**ornstein36-27B** (27B dense, Q6, reasoning) — **did not produce a deliverable.** Its thinking pass never terminates into a final answer. At the standardized `temperature 0.2` it fell into a verbatim repetition loop, emitting the same fragment (`**One Edge Case:** Set key A (update)… Correct.`) until the 16,000-token budget ran out — `finish_reason: length`, `content` empty. A diagnostic re-run at the model card's recommended `temperature 0.6 / top_p 0.95 / top_k 20` with a 24,000-token budget failed the same way: not a verbatim loop this time but runaway, self-second-guessing CoT (it even wrote candidate code *inside* the thinking channel) that still never closed `</think>`, so `content` was again empty. Two configs, ~33 min of GPU time, zero delivered code. This is a real instruction-following defect of this merge on this task, not a budget problem — an infinite loop doesn't finish with more tokens. (Note it cleared **Task B** cleanly, so the failure is task-specific, not total — see §13.7.)

**Sonnet 4.6 †** — the cleanest `Close` of the three *that delivered* (`sync.Once`, plus it wrote its own concurrent-close test); correct `container/list` LRU, correct expiry/LRU/stats; passes the neutral suite race-clean and documents `capacity==0 = unbounded`. Its **one own-suite failure is self-inflicted**: it clamps the janitor interval to a 1 s minimum, but `TestJanitorRemovesExpired` only waits 200 ms — so its own test and its own impl disagree on timing (the janitor logic itself is fine; entries also expire lazily). Net: production-correct, but the delivered `go test` is red on that one timing test — a milder version of the same "self-inconsistent test" failure mode coder-next showed.

## 13.6 Task B — Spring stack: scorecard

| Check | qwen3-coder-next | qwen36-35b-a3b | ornstein36-27B | Sonnet 4.6 † |
|---|---|---|---|---|
| Deliverable `mvn test` | ❌ **does not compile** (test sources) | ✅ **BUILD SUCCESS — 7/7** | ✅ **BUILD SUCCESS — 7/7** | ✅ **BUILD SUCCESS — 12/12** |
| — Mockito unit tests | — (compile fail) | ✅ 5/5 | ✅ 5/5 | ✅ 8/8 |
| — `@DataJpaTest` (real Hibernate+H2) | — | ✅ 2/2 (`@Version` increment) | ✅ 2/2 (`@Version` increment) | ✅ 4/4 (`@Version` increment **+ =0 on insert**) |
| **Neutral** Mockito suite vs *production* code | ✅ **5/5** | ✅ **5/5** | ✅ **5/5** (acc() adapted to 3-arg ctor) | ✅ **5/5** |
| Self-consistent (entity API matches its own tests) | ❌ test calls `Account.setId()` it never defined | ✅ `setId` + `getVersion` present | ✅ own tests compile & pass; `Account(id,owner,balance)` ctor | ✅ sets id via reflection (clean entity) |
| Mockito strict-stub discipline | n/a (didn't compile) | ✅ documented + correct | ✅ documented + correct | ✅ correct (AssertJ + ArgumentCaptor) |
| Spec adherence | Spring Boot 3.3.2, skipped file labels | Spring Boot 3.3.5, all files + assumptions | Spring Boot 3.3.0, all files + assumptions | Spring Boot 3.3.4, all files + assumptions |
| Output | 3,034 tokens, no reasoning, 86 s | 8,408 tokens (reasoning), 199 s | 10,013 tokens (reasoning), 429 s | ~13,500 tokens |

**Winner: `Sonnet 4.6` / `qwen36-35b-a3b`** — both build and pass; Sonnet's suite is the most thorough (12 tests, incl. version-on-insert, null-amount, both unknown-account directions). **`ornstein36-27B` also builds green (7/7) with correct logic** — a strong showing that makes its Task A no-show all the more striking. `qwen3-coder-next` is the only one that doesn't compile.

## 13.7 Task B: what happened

Both models' **business logic is correct** — the identical neutral Mockito suite passes **5/5 against each model's production code** (transfer, insufficient-funds, unknown-account, non-positive, same-account).

The difference is the *delivered artifact*:

- **qwen3-coder-next** wrote sound production code (idiomatic dirty-checking via `@Transactional`, `BigDecimal.compareTo`, proper custom exceptions) but its **own test sources don't compile**, for two self-inconsistencies: `assertThrows` used with no `import static org.junit.jupiter.api.Assertions.*`, and `Account` has no `setId(...)` although its test calls it. Since the task's acceptance criterion is "tests pass with `mvn test`", it fails as delivered.
- **qwen36-35b-a3b** built and passed everything, kept entity and tests consistent, used **explicit `save()`** (so persistence is verifiable with `verify(repo).save(...)`), exercised real Hibernate via `@DataJpaTest` including the `@Version` increment, and **explicitly anticipated the traps** in its assumptions (BigDecimal+`compareTo`, `@Version` as `Integer`, *"stub only methods actually invoked"* for strict stubs).

- **Sonnet 4.6 †** also built green (12/12) and was the **most thorough** — it kept the entity clean (no `setId`) and set the id via **reflection** in the unit test so the test still compiles, added a `@Version`-is-0-on-insert assertion and a null-amount case, and used AssertJ + `ArgumentCaptor`. Neutral suite 5/5.

- **ornstein36-27B** — here it **delivered**, and well: `mvn test` BUILD SUCCESS 7/7 (5 Mockito + 2 `@DataJpaTest`), real Hibernate on H2 with the `@Version` increment asserted across a flush/round-trip, `BigDecimal.compareTo` throughout, explicit `save(from)`/`save(to)`, strict-stub-clean unit tests, and a genuinely good assumptions write-up (it explicitly called out strict stubs, `@Version`, and BigDecimal scale). Its only deviation: an `Account(Long id, String owner, BigDecimal balance)` constructor instead of the 2-arg one the neutral suite assumes — a valid choice the prompt didn't pin down, so the neutral `acc()` helper was adapted by one line (passing `null` id), exactly the kind of entity-shape accommodation Sonnet's reflection trick handles. Neutral suite then 5/5. Slowest to produce it (429 s at ~19 t/s), but correct.

Same shape as Task A for coder-next: fast with correct core logic but a non-compiling/self-inconsistent test artifact; qwen36, Sonnet **and ornstein** reason through it and ship green builds, at ~3–4× the tokens. The outlier is **ornstein's split personality across tasks**: a clean, careful Spring deliverable here, yet a total non-delivery on the Go task (§13.5) — same model, same sampling, same session.

## 13.8 Recommendation (across both tasks)

- **Correctness-critical / edge-case-heavy delivery, local/offline** → `qwen36-35b-a3b`. It shipped building, passing, self-consistent code on both tasks. Accept the latency and token cost.
- **Best raw quality if a frontier API is acceptable** → `Sonnet 4.6 †` produced the most thorough Java suite and the most robust Go `Close`; its only blemish was one self-inconsistent Go timing test.
- **High-throughput / interactive coding, scaffolding, refactors** → `qwen3-coder-next` for speed — but **add a compile/test pass**: on both tasks its production logic was right while its delivered tests were broken.
- **`ornstein36-27B` → not recommended as a coding workhorse.** It produced an excellent Spring deliverable but failed to deliver the Go task at all (non-terminating reasoning, at two sampling configs), and it is the slowest model on the box (dense, ~19 t/s even with MTP). The Task-A loop is a reliability red flag for unattended/agentic use — a model that can silently burn a full token budget without emitting an answer is hard to depend on. Keep it for cases where you specifically want this merge's style on prose-shaped or framework-scaffolding work, with a budget cap and an output check.

This mirrors the throughput data on [page 8](08-performance-tuning.md): qwen36 is faster per token but spends far more tokens (reasoning); coder-next is the cheaper path when its answer is correct — and a quick `go test`/`mvn test` catches the cases where it isn't; ornstein is the slowest and the least predictable. Reserve `Sonnet 4.6` for when output quality matters more than keeping inference on-box.

## 13.9 Caveats

- **Single sample per model per task** at `temperature 0.2` — indicative, not statistical. A firm ranking needs 3–5 samples each (page 12 §12.9). This matters most for `ornstein36-27B`'s Task-A failure: it reproduced across two sampling configs (0.2 and the card's 0.6) and is a deterministic-looking loop rather than unlucky variance, but a single prompt is still one data point.
- Two tasks already reduce single-task bias. Three of four models **agree** across tasks; `ornstein36-27B` is the counterexample (clean Task B, no-delivery Task A) — the clearest argument in this whole evaluation for running **more than one task** before trusting a model.
- Both tasks reward careful lifecycle/edge-case reasoning, which favours the reasoning models. A different shape of task (algorithmic puzzle, large refactor, debugging) could shift the result.
- Cross-task consistency is the strongest signal here: the *same* failure mode (correct logic, broken delivered tests) showed up for coder-next in both Go and Java — and, more mildly, for Sonnet on Go. The opposite — a model that aces one task and fails to even respond on another — is exactly what ornstein demonstrates.
- **† Sonnet 4.6 is not apples-to-apples** (see §13.1): different harness/system prompt and default sampling vs the local models' `temperature 0.2` chat completion, and it is a frontier API model, not a local GGUF. Same task and verdict, but treat the cross-vendor ranking as indicative.

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
