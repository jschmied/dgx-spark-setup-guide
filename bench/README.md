# Coding benchmark harness

Reusable, version-controlled implementation of the standardized coding test
([page 12](../12-model-coding-test.md)); results are written up on
[page 13](../13-model-evaluation.md). Lives in the repo so it survives reboots
(the previous `/tmp` copy did not).

## What it does

Two objective, build-or-fail tasks per model:

- **Task A — Go concurrency** (`run-go.sh`): a generic thread-safe TTL+LRU cache.
  Runs `go vet`, the model's **own** `go test -race`, and a **neutral** suite.
- **Task B — Spring Boot stack** (`run-java.sh`): a money-transfer module on
  Spring Boot 3.3 / Hibernate / Mockito / JUnit 5. Runs `mvn test` (own) and a
  **neutral** Mockito suite against the model's production code.

## Sampling policy

Each model runs at its **own recommended sampling** (pinned in
`/etc/llama-server/models.ini` and the model card / GGUF `general.sampling`),
defined in `sampling_for()` in `lib.sh` — **not** a single fixed temperature.
A fixed temperature is fair for comparison but misrepresents models whose
architecture expects a different operating point (Gemma 4 and the dense Ornstein
loop at low temp; they deliver at temp ≈ 1.0).

Current map (`temp / top_p / top_k / repeat_penalty`):

| Model | temp | top_p | top_k | repeat |
|---|---|---|---|---|
| `qwen3-coder-next`   | 0.7 | 0.8  | 20 | 1.05 |
| `qwen36-35b-a3b`     | 0.6 | 0.95 | 20 | —    |
| `ornstein36-27B`     | 1.0 | 0.95 | 20 | —    |
| `ornstein36-35b-a3b` | 0.6 | 0.95 | 20 | —    |
| `gemma-4-26B-A4B`    | 1.0 | 0.95 | 64 | —    |

## Prerequisites

- Router running on `http://127.0.0.1:8080` with the model in the preset.
- `jq`, Go 1.22+, JDK 17 + Maven (`sudo apt-get install -y openjdk-17-jdk maven`).
- Pre-warm `~/.m2` once (a throwaway Spring Boot 3.3 project + `mvn test`) so
  per-model Java builds are fast and comparable.
- API key: set `LLAMA_API_KEY`, or run where the first line of
  `/etc/llama-server/api_keys.txt` is readable (the scripts never print it).

## Usage

```bash
cd bench
./run-all.sh                       # all known models, both tasks
./run-all.sh gemma-4-26B-A4B       # one or more specific models
./run-go.sh   gemma-4-26B-A4B      # just Task A
./run-java.sh gemma-4-26B-A4B      # just Task B
```

Override sampling for an unknown model with `BENCH_TEMP/BENCH_TOPP/BENCH_TOPK`,
or the host with `LLAMA_HOST`.

## Layout

```
prompts/cache.txt              Task A prompt (verbatim from page 12.2)
prompts/transfer.txt           Task B prompt (verbatim from page 12.8.2)
neutral/cache_neutral_test.go  Task A neutral suite
neutral/NeutralTransferTest.java  Task B neutral suite
extract_java.py                multi-file Maven/Java extractor
lib.sh                         API key + per-model sampling + call_model
run-go.sh / run-java.sh        single-task runners (print a VERDICT line)
run-all.sh                     driver
results/   (gitignored)        raw model JSON responses
.work/     (gitignored)        extracted projects + build output
```

Raw responses (`results/`) and build trees (`.work/`) are gitignored — they are
regenerable artifacts, not source.
