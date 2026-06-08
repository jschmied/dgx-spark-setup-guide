# 12. Standardized coding test for served models

[← Security checklist](11-security-checklist.md) · [Index](README.md) · [Next: Model evaluation →](13-model-evaluation.md)

A reproducible way to compare the **coding ability** of any model the router serves, with an **objective** pass/fail — the code either builds and passes its tests, or it doesn't. Use it whenever you add a model to the preset. Results are on [page 13](13-model-evaluation.md).

There are **two tasks**, run the same way (identical sampling → extract the files → build → test):

- **Task A — Go concurrency** (§12.1–12.7): a generic, thread-safe TTL+LRU cache. `go test -race` gives an objective verdict on the hardest requirement, concurrency.
- **Task B — Spring Boot stack** (§12.8): a Spring Boot + Hibernate/JPA + Mockito + JUnit 5 money-transfer module; `mvn test` is the verdict. Exercises framework wiring, ORM/persistence, and mocking discipline (Mockito strict stubs).

Run both — a single task can flatter one model. The general caveats in §12.9 apply to both.

## 12.1 Prerequisites

- The router is running with the model defined in `models.ini` (pages 6/8). Adding a model = a new section in the preset, then `sudo systemctl restart llama-router`.
- A Go toolchain on the box: `go version` (1.22+).
- `jq` and an API key from `/etc/llama-server/api_keys.txt`.

Throughout, set the model id under test:

```bash
MODEL=qwen36-35b-a3b      # <- the preset section / model id to evaluate
```

> **Swap latency.** With `--models-max 1` the first request swaps the model in (tens of seconds for a multi-GB GGUF). That's normal; the timing you care about is reported by the server, not the wall clock of the first call.

## 12.2 The test prompt

````bash
cat > /tmp/cprompt.txt <<'PROMPT'
Implement a production-quality, generic, thread-safe in-memory cache in Go (1.22+), as a single cache.go plus cache_test.go. No third-party dependencies.

Public API — match exactly:
type Cache[K comparable, V any] struct { /* ... */ }
func New[K comparable, V any](capacity int, defaultTTL time.Duration) *Cache[K, V]
func (c *Cache[K, V]) Get(key K) (V, bool)
func (c *Cache[K, V]) Set(key K, value V)
func (c *Cache[K, V]) SetWithTTL(key K, value V, ttl time.Duration)
func (c *Cache[K, V]) Delete(key K)
func (c *Cache[K, V]) Len() int
func (c *Cache[K, V]) Stats() (hits, misses, evictions uint64)
func (c *Cache[K, V]) Close()

Semantics — follow precisely:
1. capacity is the max number of live entries. Adding a new key beyond capacity evicts the least-recently-used live entry and increments evictions.
2. defaultTTL <= 0 means entries never expire by default. In SetWithTTL, ttl <= 0 means that entry never expires; ttl > 0 means absolute expiry at now+ttl.
3. Get on a missing OR expired key returns the zero value + false, counts a miss, and must NOT promote LRU and must NOT return the expired value. Get on a live key returns value + true, counts a hit, and promotes it to most-recently-used.
4. Set/SetWithTTL on an existing key updates the value, resets its TTL, and promotes to MRU (not an eviction).
5. Expired entries are reclaimed two ways: lazily on access, AND by a background janitor goroutine on a sensible interval. Close() stops the janitor, is safe to call multiple times (subsequent calls are no-ops), and leaks no goroutine.
6. All operations are safe under concurrent use by many goroutines; Get/Set are O(1).

Deliverables:
- cache.go and cache_test.go.
- Tests covering: basic get/set, LRU eviction order, TTL lazy expiry, janitor expiry, update-resets-TTL-and-LRU, stats correctness, capacity == 0 behavior, and a concurrency test that passes under `go test -race`.
- State your assumptions and the time/space complexity.

Idiomatic Go, no busy-waiting, minimal lock contention, no goroutine leaks. Put cache.go in one fenced ```go code block and cache_test.go in a separate fenced ```go code block, each immediately preceded by a line naming the file.
PROMPT
````

> Known ambiguity (leave it in — it's a useful discriminator): `capacity == 0` is underspecified. Strong models *document* their choice (e.g. "0 = unlimited" vs "0 = store nothing"). The neutral suite in 12.5 avoids asserting on it.

## 12.3 Run the prompt against a model

Identical sampling for every model so it's about capability, not luck. Non-streaming, generous token budget (reasoning models spend thousands of tokens thinking before emitting code).

```bash
KEY=$(sudo head -1 /etc/llama-server/api_keys.txt)

body=$(jq -n --rawfile p /tmp/cprompt.txt --arg m "$MODEL" \
  '{model:$m, messages:[{role:"user",content:$p}],
    temperature:0.2, top_p:0.9, max_tokens:16000, stream:false}')

curl -s --max-time 1800 http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "$body" > /tmp/out_$MODEL.json

# sanity: did it finish cleanly (not hit the token cap)?
jq -r '{finish:.choices[0].finish_reason,
        completion_tokens:.usage.completion_tokens,
        has_reasoning:((.choices[0].message.reasoning_content//"")|length>0),
        content_len:(.choices[0].message.content|length)}' /tmp/out_$MODEL.json
```

`finish` must be `stop`. If it's `length`, raise `max_tokens` and re-run — a truncated answer isn't a fair sample.

## 12.4 Extract the two Go files

The prompt asks for `cache.go` then `cache_test.go`, each in its own fenced block. This awk pulls the N-th fenced block regardless of language tag:

````bash
jq -r '.choices[0].message.content' /tmp/out_$MODEL.json > /tmp/$MODEL.md
block() { awk -v n="$1" '/^```/{f=!f; if(f)c++; next} f&&c==n' /tmp/$MODEL.md; }

DIR=/tmp/test_$MODEL; mkdir -p "$DIR"
block 1 > "$DIR/cache.go"
block 2 > "$DIR/cache_test.go"
printf 'module cachetest\n\ngo 1.22\n' > "$DIR/go.mod"

# spot-check: both should be `package cache`, no stray ``` lines
grep -m1 '^package' "$DIR"/cache.go "$DIR"/cache_test.go
grep -c '```' "$DIR"/cache.go "$DIR"/cache_test.go   # expect 0 and 0
````

If a model wrapped both files in one block, or put prose between them, fix the block numbers by eye — list the fence lines with `grep -nE` for code-fence markers in `/tmp/$MODEL.md` and adjust the `block N` arguments.

## 12.5 Objective checks: vet, own tests, neutral tests

```bash
cd /tmp/test_$MODEL
export GOFLAGS=-mod=mod GOCACHE=/tmp/gocache GOPATH=/tmp/gopath

go vet ./...                              # must be clean
go test -race -count=1 -timeout 120s ./... # the model's OWN suite
```

A model passing its **own** tests is only meaningful if those tests are rigorous, so also run a **neutral** suite that checks the unambiguous requirements (idempotent `Close`, expired→miss, LRU order, race-safety) and asserts nothing on the ambiguous `capacity==0`. Drop the model's `cache.go` next to it:

```bash
cat > /tmp/neutral_test.go <<'EOF'
package cache

import (
	"sync"
	"testing"
	"time"
)

// Rule 5: Close must be safe to call multiple times (no panic).
func TestNeutralCloseIdempotent(t *testing.T) {
	c := New[string, int](10, time.Minute)
	c.Close()
	c.Close()
	c.Close()
}

// Rule 3: Get on an expired key returns false.
func TestNeutralExpiredMiss(t *testing.T) {
	c := New[string, int](10, time.Minute)
	defer c.Close()
	c.SetWithTTL("k", 1, 30*time.Millisecond)
	time.Sleep(80 * time.Millisecond)
	if v, ok := c.Get("k"); ok {
		t.Fatalf("expired key returned (%v,%v), want miss", v, ok)
	}
}

// Rule 1: at capacity, the least-recently-used live entry is evicted.
func TestNeutralLRUOrder(t *testing.T) {
	c := New[string, int](2, time.Minute)
	defer c.Close()
	c.Set("a", 1)
	c.Set("b", 2)
	if _, ok := c.Get("a"); !ok {
		t.Fatal("a should be live")
	} // a now MRU
	c.Set("c", 3) // capacity 2 -> evict LRU which is b
	if _, ok := c.Get("b"); ok {
		t.Fatal("b should have been evicted (LRU)")
	}
	if _, ok := c.Get("a"); !ok {
		t.Fatal("a should still be present")
	}
	if _, ok := c.Get("c"); !ok {
		t.Fatal("c should be present")
	}
}

// Rule 6: concurrency-safe (run under -race).
func TestNeutralRace(t *testing.T) {
	c := New[int, int](128, 50*time.Millisecond)
	defer c.Close()
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(base int) {
			defer wg.Done()
			for i := 0; i < 2000; i++ {
				k := (base*2000 + i) % 256
				c.Set(k, i)
				c.Get(k)
			}
		}(g)
	}
	wg.Wait()
}
EOF

d=/tmp/neutral_$MODEL; mkdir -p "$d"; cd "$d"
printf 'module cachetest\n\ngo 1.22\n' > go.mod
cp /tmp/test_$MODEL/cache.go .; cp /tmp/neutral_test.go .
go test -race -count=1 -timeout 60s -v ./...
```

The neutral suite is the fair judge: it depends only on the public API and the unambiguous rules.

## 12.6 Optional: cross-testing to neutralize test-quality bias

Run a **reference** model's test suite against the new model's implementation (and vice versa). Disagreements isolate three things:
- a real bug in the implementation,
- a buggy assertion in the *test* suite (if **both** implementations contradict the same assertion, the test is wrong, not the code),
- a legitimate spec-interpretation difference (e.g. `capacity==0`).

```bash
mkdir -p /tmp/xcross && cd /tmp/xcross && printf 'module cachetest\n\ngo 1.22\n' > go.mod
cp /tmp/test_$MODEL/cache.go .            # new model's implementation
cp /tmp/test_REFERENCE_MODEL/cache_test.go .   # a trusted model's tests
go test -race -count=1 -timeout 120s ./...
```

## 12.7 Scoring rubric

| Check | Strong answer |
|---|---|
| Compiles + `go vet` clean | required baseline |
| Neutral suite `go test -race` | **all pass** — this is the headline result |
| Data structure | `map` + doubly-linked list (`container/list` or hand-rolled) → true O(1) LRU; a slice scan is O(n) = weak |
| `Close()` idempotency | `sync.Once`, a closed-channel guard, or mutex+bool — must not panic on a second call |
| Expired read | `Get` on expired returns miss **without** promoting LRU |
| `capacity==0` | any behavior, **documented** |
| Deliverables | both files **plus** the assumptions + complexity write-up |
| Tests | exercise eviction *order* and janitor timing, with **correct** assertions |
| Cost | tokens + wall time (reasoning models cost far more; weigh against correctness) |

## 12.8 Task B: Spring Boot / Hibernate / Mockito / JUnit 5 (Java)

A realistic framework task: a money-transfer module across the named stack. The verdict is `mvn test` (BUILD SUCCESS with tests actually run). It probes things the Go task doesn't — JPA/Hibernate mapping, `@Transactional`/persistence semantics, `BigDecimal` money handling, and **Mockito strict-stub discipline**.

### 12.8.1 Prerequisites

- JDK 17+ and Maven (the box may ship only a JRE):
  ```bash
  sudo apt-get install -y openjdk-17-jdk maven
  export JAVA_HOME=$(ls -d /usr/lib/jvm/java-17-openjdk-* | head -1)
  ```
- Maven Central reachable. **Pre-warm the cache once** so model builds are fast and comparable — a throwaway minimal Spring Boot 3.3 project (parent `spring-boot-starter-parent:3.3.5`, deps `spring-boot-starter-data-jpa`, `spring-boot-starter-test`, `com.h2database:h2`) with one trivial `@Test`, then `JAVA_HOME=$JAVA_HOME mvn -q test`. This downloads the whole dependency tree to `~/.m2` up front.

### 12.8.2 The prompt

````bash
cat > /tmp/jprompt.txt <<'PROMPT'
Build a small but production-quality Spring Boot (3.3.x, Java 17) money-transfer module as a single Maven project that compiles and whose tests pass with `mvn test`. No code outside the stack: Spring Boot, Spring Data JPA / Hibernate, H2 (test runtime), JUnit 5, Mockito. Use base package com.example.bank.

Domain & rules:
- JPA @Entity Account: Long id (generated), String owner (not null), BigDecimal balance (not null), and a @Version field for optimistic locking. Money is always BigDecimal — never double.
- Spring Data interface AccountRepository extends JpaRepository<Account, Long> with a derived query List<Account> findByOwner(String owner).
- TransferService with a @Transactional method: void transfer(Long fromId, Long toId, BigDecimal amount). Rules, each enforced:
  1. amount must be non-null and strictly positive, else throw IllegalArgumentException.
  2. fromId and toId must differ, else IllegalArgumentException.
  3. both accounts must exist, else throw a custom AccountNotFoundException (extends RuntimeException) naming the missing id.
  4. the source balance must be >= amount, else throw a custom InsufficientFundsException (extends RuntimeException).
  5. on success, debit the source and credit the destination and persist both. Use BigDecimal.compareTo for comparisons, not equals.

Deliverables (each file in its OWN fenced ```java (or ```xml) code block, immediately preceded by a line giving its path):
- pom.xml — single project, Spring Boot 3.3.x parent, starters spring-boot-starter-data-jpa and spring-boot-starter-test, H2 with test scope, Java 17.
- src/main/java/com/example/bank/BankApplication.java — @SpringBootApplication.
- src/main/java/com/example/bank/Account.java
- src/main/java/com/example/bank/AccountRepository.java
- src/main/java/com/example/bank/TransferService.java
- src/main/java/com/example/bank/AccountNotFoundException.java
- src/main/java/com/example/bank/InsufficientFundsException.java
- src/test/java/com/example/bank/TransferServiceTest.java — pure unit test with Mockito: @ExtendWith(MockitoExtension.class), @Mock the repository, @InjectMocks the service. Cover: successful transfer updates both balances and persists; insufficient funds throws and persists nothing; unknown account throws; non-positive amount throws; same-account throws. Verify repository interactions and assert no unnecessary stubbing (keep it compatible with Mockito strict stubs).
- src/test/java/com/example/bank/AccountRepositoryDataJpaTest.java — a @DataJpaTest integration test running real Hibernate on H2: persist accounts and assert findByOwner returns the right rows, and assert the @Version value increments after an update+flush.

Constraints: must build and pass with `mvn test` offline-friendly (only the named dependencies). Idiomatic Spring. State any assumptions briefly at the end.
PROMPT
````

### 12.8.3 Run, then extract the multi-file project

Run exactly as §12.3 but with `/tmp/jprompt.txt` and a larger budget (Spring projects are bigger): `max_tokens: 20000`. Confirm `finish == stop`.

Models often **skip the per-file path labels**, so derive each filename from the block's content — the `xml` block is `pom.xml`; each `java` block's `package` + public type name (+ a `Test`/`IT` suffix → test source) gives its path:

````bash
cat > /tmp/extract.py <<'PY'
import json, re, sys, os
content = json.load(open(sys.argv[1]))["choices"][0]["message"]["content"]
lines = content.split("\n"); blocks=[]; i=0
while i < len(lines):
    s = lines[i].lstrip()
    if s.startswith("```"):
        lang = s[3:].strip().lower(); body=[]; i+=1
        while i < len(lines) and not lines[i].lstrip().startswith("```"):
            body.append(lines[i]); i+=1
        i+=1; blocks.append((lang, "\n".join(body)))
    else: i+=1
pkgre=re.compile(r'^\s*package\s+([\w.]+)\s*;', re.M)
typere=re.compile(r'(?:public\s+|final\s+|abstract\s+)*\b(class|interface|enum|record)\s+(\w+)')
outdir=sys.argv[2]
for lang, body in blocks:
    if 'xml' in lang or '<project' in body[:200]:
        path="pom.xml"
    elif 'java' in lang or 'package ' in body[:200]:
        m=pkgre.search(body); pkg=m.group(1) if m else "com.example.bank"
        t=typere.search(body)
        if not t: continue
        name=t.group(2)
        kind="test" if re.search(r'(Test|Tests|IT)$', name) else "main"
        path=f"src/{kind}/java/{pkg.replace('.','/')}/{name}.java"
    else: continue
    full=os.path.join(outdir, path); os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full,"w").write(body.rstrip()+"\n"); print("wrote", path)
PY
python3 /tmp/extract.py /tmp/out_$MODEL.json /tmp/proj_$MODEL
find /tmp/proj_$MODEL -type f | sort
````

### 12.8.4 Build and test (the verdict)

```bash
export JAVA_HOME=$(ls -d /usr/lib/jvm/java-17-openjdk-* | head -1)
cd /tmp/proj_$MODEL
mvn -B -q test                       # BUILD SUCCESS, and tests must actually run
# per-class counts:
grep -H "Tests run" target/surefire-reports/*.txt
```

A model that ships a project which **doesn't compile** (e.g. a test that calls `Account.setId(...)` the entity never defines, or `assertThrows` without the JUnit 5 static import) fails here regardless of how good the production code is.

### 12.8.5 Neutral suite — judge the business logic separately

A model's own tests may not compile or may be lax, so run an identical **neutral** Mockito suite against each model's *production* code (copy `src/main`, drop in the neutral test, build). It asserts only the unambiguous business rules and is written to be Mockito-strict-safe (stub only what's used; failure paths don't stub the lookups they never reach):

```bash
cat > /tmp/NeutralTransferTest.java <<'EOF'
package com.example.bank;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.math.BigDecimal;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class NeutralTransferTest {
    @Mock AccountRepository repo;
    @InjectMocks TransferService service;

    private Account acc(String owner, String bal) { return new Account(owner, new BigDecimal(bal)); }

    @Test void transfersFunds() {
        Account from = acc("Alice", "100.00"), to = acc("Bob", "50.00");
        when(repo.findById(1L)).thenReturn(Optional.of(from));
        when(repo.findById(2L)).thenReturn(Optional.of(to));
        service.transfer(1L, 2L, new BigDecimal("30.00"));
        assertEquals(0, from.getBalance().compareTo(new BigDecimal("70.00")));
        assertEquals(0, to.getBalance().compareTo(new BigDecimal("80.00")));
    }

    @Test void insufficientFundsThrowsAndKeepsBalances() {
        Account from = acc("Alice", "20.00"), to = acc("Bob", "50.00");
        when(repo.findById(1L)).thenReturn(Optional.of(from));
        when(repo.findById(2L)).thenReturn(Optional.of(to));
        assertThrows(InsufficientFundsException.class, () -> service.transfer(1L, 2L, new BigDecimal("30.00")));
        assertEquals(0, from.getBalance().compareTo(new BigDecimal("20.00")));
        assertEquals(0, to.getBalance().compareTo(new BigDecimal("50.00")));
    }

    @Test void unknownAccountThrows() {
        when(repo.findById(1L)).thenReturn(Optional.of(acc("Alice", "100.00")));
        when(repo.findById(2L)).thenReturn(Optional.empty());
        assertThrows(AccountNotFoundException.class, () -> service.transfer(1L, 2L, new BigDecimal("10.00")));
    }

    @Test void nonPositiveAmountThrows() {
        assertThrows(IllegalArgumentException.class, () -> service.transfer(1L, 2L, new BigDecimal("0.00")));
        assertThrows(IllegalArgumentException.class, () -> service.transfer(1L, 2L, new BigDecimal("-5.00")));
    }

    @Test void sameAccountThrows() {
        assertThrows(IllegalArgumentException.class, () -> service.transfer(1L, 1L, new BigDecimal("10.00")));
    }
}
EOF

d=/tmp/neutral_$MODEL; rm -rf "$d"; mkdir -p "$d/src/test/java/com/example/bank"
cp /tmp/proj_$MODEL/pom.xml "$d/pom.xml"; cp -r /tmp/proj_$MODEL/src/main "$d/src/main"
cp /tmp/NeutralTransferTest.java "$d/src/test/java/com/example/bank/NeutralTransferTest.java"
( cd "$d" && JAVA_HOME=$(ls -d /usr/lib/jvm/java-17-openjdk-* | head -1) mvn -B -q test )
```

It mocks `AccountRepository`, builds accounts via the constructor, stubs `findById`, and asserts: transfer debits/credits correctly; `InsufficientFundsException` leaves balances unchanged; `AccountNotFoundException` on a missing id; `IllegalArgumentException` on non-positive amount and on same-account. It is strict-stub-safe (the no-lookup paths stub nothing) and passes against either a dirty-checking service (no explicit `save`) or one that calls `save()`, because it asserts on balances/exceptions, not on `save` interactions.

### 12.8.6 What to look for

| Check | Strong answer |
|---|---|
| `mvn test` | **BUILD SUCCESS**, tests run > 0 — the headline |
| Self-consistency | entity API matches its own tests (a `setId` the test calls must exist) |
| Money | `BigDecimal` everywhere, compared with `compareTo` (never `double`/`equals`) |
| Optimistic locking | `@Version` present **and** asserted (increments after update+flush) |
| Mockito discipline | no `UnnecessaryStubbingException`; failure paths use `verify(..., never())` |
| Real ORM | `@DataJpaTest` slice on H2 exercises actual Hibernate, not just mocks |
| Deliverables | all files **plus** the assumptions write-up |

## 12.9 Caveats — making it robust

- **One sample is indicative, not statistical.** For a firm ranking run 3–5 samples per model (`-count` of *runs*, not test iterations) and report the spread; low temperature reduces but doesn't remove variance.
- **Keep sampling identical across models** (the WebUI would otherwise inject its own params — see page 6 notes). The preset's per-model sampling is overridden by what you send here.
- **Reasoning models** (e.g. Qwen3.6) emit a long thinking pass first — budget tokens and time accordingly, and compare cost honestly.
- **Run both tasks** (Task A §12.1–12.7 *and* Task B §12.8) — a single task can flatter one model. Adding a third task type (a parser, a data pipeline) cross-checks further.
- **Task B needs network** (Maven Central) on the first build and a JDK 17+/Maven install; pre-warm `~/.m2` once so per-model builds are fast and comparable.

---

[← Security checklist](11-security-checklist.md) · [Index](README.md) · [Next: Model evaluation →](13-model-evaluation.md)
