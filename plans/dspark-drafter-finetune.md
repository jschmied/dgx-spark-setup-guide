# Plan: fine-tune a DSpark drafter on our own agent traces

**Status: CLOSED at Gate 0 — 2026-08-17.** Phase 0 ran; DSpark is **27 % slower than MTP** on real
agent traffic and the reasoning behind the whole plan was refuted. Kept as the record of why, and
because the method and the tooling stay valid if a better drafter appears.

> ## Gate 0 result — FAIL
>
> 24 real agent contexts replayed from our SWE-bench trajectories (2 k–43 k tokens, median 10.7 k),
> paired, `usage.completion_tokens`:
>
> | | MTP n=3 | DSpark k=7 |
> |---|---|---|
> | median | **26.4 tok/s** | 18.9 tok/s |
> | mean | 26.3 | 20.1 |
> | wins | — | **3 / 24** |
> | tok per SSE delta (acceptance proxy) | 5.81 | 5.54 |
> | median TTFT | 7.74 s | 7.39 s |
>
> Gate required ≥ +25 % median per-prompt. **Measured −27.1 %.**
>
> **The workload hypothesis is dead.** The plan assumed DSpark underperformed on synthetic prompts
> because they lack echoed content, and would recover on edit-heavy agent traffic. Then the deficit
> should shrink as context grows. It does not:
>
> ```
>   < 8k   n=9   median −27.4 %   wins 1/9
>  8–20k   n=8   median −24.3 %   wins 1/8
>   >20k   n=7   median −27.4 %   wins 1/7
> ```
>
> Flat across two orders of magnitude. Acceptance confirms it: DSpark drafts 7 tokens and lands
> **fewer** than MTP's 3. TTFT is unchanged, so this is pure decode deficit, not a prefill trade.
>
> **Consequence for the fine-tune:** we would not be adapting a good drafter to our domain, we would
> be repairing a starting point that is worse than the MTP head already shipped inside the target —
> at ~19 h of training, with A5 (does SpecForge accept an NVFP4 target?) still unanswered, and at a
> capacity cost of 7.18 → 1.93 concurrent full-context sessions.
>
> **Decision: stay on MTP n=3. Do not train.**
>
> **The one thread still worth pulling** is A-quant: this drafter was trained against
> `Qwen3.8-27B-FP8`. Whether that explains the deficit is testable by serving the FP8 target — no
> training required. Cheap, and it would settle whether *any* DSpark drafter can work here.

**Target:** `unsloth/Qwen3.8-27B-NVFP4` on GB10 · **Written:** 2026-08-17

---

## 1. What this is for

We serve Qwen3.8-27B-NVFP4 with MTP `nspec=3`. A third-party DSpark drafter
(`RadixArk/Qwen3.8-27B-DSpark`) runs and gives **+12 % mean** on a synthetic probe — far below its
published figures, and it regresses on prose. Two explanations are open:

1. **Workload mismatch** — published numbers come from an *edit-heavy* harness; our probe generates
   fresh short text.
2. **Quantization mismatch** — it was trained against `Qwen/Qwen3.8-27B-FP8`, and we serve the
   Unsloth NVFP4 checkpoint, so its layer taps see activations it was never fitted to.

A fine-tune on our own SWE-bench agent traces would address both at once, and would replace an
undocumented training corpus with one we control.

**This plan exists to decide whether that is worth doing — and to stop cheaply if it is not.**

---

## 2. Recommendation up front

**Do Phase 0 only. Decide the rest afterwards.**

Phase 0 costs a few hours and answers the question that makes every later phase either worthwhile
or pointless: *does the existing drafter help on real agent traffic?* Training from scratch is
rejected outright (§7). A fine-tune is plausible but rests on four unverified assumptions (§6),
three of which Phase 0 and Phase 2 settle for almost nothing.

---

## 3. What we already measured

Paired, `usage.completion_tokens` (never stream deltas), 3 runs, util 0.6, `--max-model-len 32768`:

| prompt | MTP n=3 | DSpark k=7 | DSpark k=14 |
|---|---|---|---|
| P1 code (py) | 30.3 | **43.8 (+45 %)** | 42.8 |
| P2 prose | 18.0 | 16.4 (−9 %) | 13.8 (−23 %) |
| P3 code (bash) | 25.3 | 23.4 (−8 %) | 19.9 (−21 %) |
| **mean** | 24.9 | **28.0 (+12 %)** | 26.0 |

`k=14` is worse than `k=7` here, contradicting the published recommendation. The reported
72–75 tok/s single-stream did not reproduce; our best single figure is 43.8 on one code prompt.

**Corpus we hold today** (100 SWE-bench Multilingual trajectories):

| | |
|---|---|
| trajectories | 100 |
| assistant turns | 3,806 |
| assistant tokens (est.) | **1.34 M** — the learnable signal |
| prompt tokens (est.) | 1.52 M — forward-pass cost |
| share of a typical drafter corpus (~67 M) | **2 %** |

Estimated at 3.6 chars/token.

---

## 4. Phases

Each phase has an explicit **gate**. A gate that fails ends the plan — that is the point of having
them, and it is the part most likely to be skipped under momentum.

### Phase 0 — Is the existing drafter worth anything on real work?
**Cost:** ~3–4 h · **Blocks everything else**

The +12 % came from short synthetic prompts. Agent traffic echoes file contents constantly, which
is exactly where a drafter should win. Measure that before building anything.

1. Extract 20–30 real prompts from `/root/gen-multilingual-100/*/​*.traj.json` — full agent context,
   not the first turn. `bench_client_real.py` already does this; verify it reads `usage` tokens.
2. Serve MTP `nspec=3`, run the set, record tok/s and TTFT per prompt.
3. Restart with DSpark `k=7` (`VLLM_QWEN38_SPEC=dspark`), same prompts, same seed.
4. Compare paired, per prompt — not as a single mean, which hid the prose regression above.

**Gate 0** — proceed only if DSpark shows **≥ +25 % mean on real agent prompts**, or a clear win on
the long-context subset. Below that, a fine-tune is optimising a lever that does not move.

> If Gate 0 fails, the honest outcome is: stay on MTP, write up why, stop. That is a successful
> result, not a wasted afternoon.

### Phase 1 — Can we even train this here?
**Cost:** ~4–6 h · **Pure feasibility, no training**

Four unknowns, all cheap to settle, none of which should be assumed:

1. **Does SpecForge support DSpark at all?** It is EAGLE3-centric. The RadixArk card says the model
   was trained with SpecForge, but that does not prove the DSpark recipe is public.
2. **Does it build on aarch64/GB10?** Our source-build history on this box is poor — global OOMs,
   `BUILD_JOBS=4` mandatory. Build with the fleet stopped.
3. **Does online training accept an NVFP4 frozen target?** Trainers commonly assume a BF16/FP16
   teacher. If it needs BF16 weights, the quantization-mismatch fix evaporates — we would be
   training against the very hidden states we are trying to move away from.
4. **Will vLLM load our output?** RadixArk's checkpoint needed
   `architectures: ["Qwen3DSparkModel"]` instead of the shipped `["DSparkDraftModel"]`. Whatever
   SpecForge emits will need the same treatment.

**Gate 1** — all four answered yes, or a documented workaround exists. Item 3 is the one that can
quietly invalidate the whole rationale.

### Phase 2 — Generate training data
**Cost:** 1–3.5 days wall-clock, unattended

Measured rate: **~12 instances/h** at 6 workers.

| instances | generation | assistant tokens |
|---|---|---|
| 100 (have) | — | 1.34 M |
| ~300 (Multilingual full) | ~1 day | ~4 M |
| ~1000 | ~3.5 days | ~13.4 M |

Run with the **patched harness** (`/root/patch_docker2.py`) and `preflight-swebench.sh` — an
unpatched run discards ~39 % of submissions, and a drafter trained on truncated trajectories learns
the truncation.

**Gate 2** — triage clean: no `scaffold-discard`, no `infra-nostart`. Use `/root/triage.py`.

### Phase 3 — Train
**Cost:** ~19 h (100 traces) to ~192 h (1000 traces)

Prefer **online** training: target frozen, drafter trained alongside, hidden states generated on the
fly. It fits:

| | GB |
|---|---|
| Target NVFP4 (frozen) | 22.6 |
| Drafter bf16 (1.36 B) | 2.7 |
| Gradients | 2.7 |
| Adam m+v fp32 | 10.9 |
| Master weights fp32 | 5.4 |
| **total** | **44.3** — leaves ~77 GB of 121 |

Offline is the fallback and costs storage: **50 KB/token** (5 taps × 5120 × bf16) → 136 GB for our
100 traces, ~3.1 TB for all of SWE-bench.

Start from RadixArk's checkpoint, not from scratch. 2 epochs.

**Gate 3** — training loss decreasing and acceptance length on a held-out slice **above** the
stock drafter's. Hold out ~10 % of trajectories *by instance*, never by turn — turns within one
trajectory are heavily correlated and a per-turn split will report a flattering number.

### Phase 4 — Evaluate honestly
**Cost:** ~4 h + one SWE mini run

1. Acceptance length on held-out traces vs stock DSpark vs MTP.
2. Throughput on the Phase 0 prompt set, paired.
3. **A SWE mini run (20–30 instances)** — speculative decoding is distribution-preserving in
   theory, so quality should be unchanged; verify rather than assert.

**Gate 4** — throughput beats MTP on real prompts **and** SWE resolution is within noise of the
MTP baseline. A drafter that is faster and worse is a regression.

### Phase 5 — Deploy or shelve
Either flip `VLLM_QWEN38_SPEC=dspark` in the unit and document it, or write up the negative result.
Both are outcomes; only silence is failure.

---

## 5. Cost summary

| phase | wall-clock | attended? |
|---|---|---|
| 0 — real-prompt check | 3–4 h | yes |
| 1 — feasibility | 4–6 h | yes |
| 2 — data generation | 1–3.5 days | no |
| 3 — training | 19–192 h | no |
| 4 — evaluation | ~4 h + SWE run | partly |
| **to first go/no-go** | **~4 h** | |
| **full path, 100 traces** | **~3–4 days** | |
| **full path, 1000 traces** | **~2 weeks** | |

---

## 6. Assumptions — and how to check each one

| # | assumption | confidence | how to verify | if wrong |
|---|---|---|---|---|
| A1 | GB10 ≈ 1/5–1/17 of an H200 for this workload | **low** | time 100 training steps, extrapolate | timings in §5 are wrong by up to 3× |
| A2 | ~48 H200-h is a valid reference for a 1.36 B drafter | medium | third-party report, not ours | rescale §5 |
| A3 | SpecForge supports DSpark, not just EAGLE3 | **low** | Phase 1 | no public recipe → plan dies |
| A4 | SpecForge builds on aarch64 | low | Phase 1 | port effort, possibly large |
| A5 | Online training accepts an NVFP4 frozen target | **low** | Phase 1 | core rationale (quantization fix) is lost |
| A6 | Fine-tuning needs ≪ a from-scratch corpus | medium | Gate 3 | need 10× data → Phase 2 becomes weeks |
| A7 | Our traces represent the traffic we care about | medium | they *are* our traffic | overfits to SWE-bench specifically |
| A8 | 2 epochs suffice | low | loss curve | more epochs, linear cost |
| A9 | 3.6 chars/token | medium | tokenize a sample | token counts shift ±20 % |
| A10 | Spec-decode preserves quality, so only speed matters | high | Gate 4 measures it anyway | quality regression → abort |

**A1, A3, A5 are the load-bearing ones.** All three are low confidence, and Phase 1 settles A3–A5
for ~5 hours. Do not start Phase 2 before then.

---

## 7. Explicitly rejected

- **Training from scratch.** 10–34 days on GB10, and our corpus is 2 % of what that needs. Even all
  of SWE-bench full (8 days of generation) yields ~46 %.
- **4-bit KV cache** (`turboquant_*`) to free memory. Quality risk not worth it — decided
  2026-08-17. It would otherwise solve the DSpark-plus-262 k-context conflict.
- **`k=14`.** Measured worse than `k=7` here.
- **Raising `gpu-memory-utilization` toward 0.9.** Killed this box once (global OOM, power-cycle).
  Current 0.76 leaves ~20 GB.

---

## 8. Risks

| risk | impact | mitigation |
|---|---|---|
| Box OOM during training | power-cycle, lost run | keep util ≤ 0.76, `MAX_JOBS=2`, checkpoint often |
| Prod down for days | no serving | train on a copy at reduced context, or accept a maintenance window |
| Drafter overfits to SWE-bench | looks great, helps nothing else | hold out by instance; keep a non-SWE prompt set in Phase 4 |
| Trained against the wrong precision | rationale lost, result misleading | A5 is a Phase 1 gate, not an afterthought |
| Sunk-cost drift past a failed gate | weeks spent on a dead lever | gates are written down here *before* starting |

---

## 9. Open questions for us

1. **Is a domain drafter the right lever at all?** Phase 0 tells us the ceiling. If real-work gain
   is ~10 %, prefill/TTFT work may be worth more — at 32 k context TTFT is ~100 s and is the
   dominant cost in agent loops.
2. **How much prod downtime is acceptable?** Phase 3 occupies the GPU for a day or more.
3. **Do we need our own drafter, or just a better-matched one?** An FP8-target drafter against an
   FP8 target would test the mismatch hypothesis without any training.
4. **Is 1000 instances of SWE-bench the right corpus,** or should we mix in real usage from the
   router logs?

---

## 10. First concrete step

```bash
# Phase 0, step 1 — build the real-prompt set from existing trajectories
ssh root@10.0.0.8 '/root/sweb-venv/bin/python - <<EOF
import json, glob
# extract N full agent contexts, mid-trajectory (not turn 0)
EOF'
```

Nothing before Gate 0.
