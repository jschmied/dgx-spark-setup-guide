# Plan: give the MTP head one copy per draft position

**Status: Phase 0 RUN 2026-08-17 — wiring works, but two of this plan's premises were wrong.**
Revised below; still viable, but more expensive and less certain than first written.

> ## Phase 0 result
>
> Duplicated `mtp.layers.0.*` to `.layers.1.*` byte-identically, `mtp_num_hidden_layers: 2`, served,
> measured on the same 8 agent contexts.
>
> | | 1 head (baseline) | 2 identical heads |
> |---|---|---|
> | pos 0 | 87.2 % | 86.1 % |
> | pos 1 | 76.7 % | **72.1 %** |
> | pos 2 | 67.2 % | 64.2 % |
> | overall | 77.0 % | 74.1 % |
> | tokens/step | 3.31 | 3.22 |
>
> **The difference is real, not noise.** A repeat of the 1-head baseline reproduced *bit-identically*
> (335 steps, 1005 drafted, 774 accepted) — vLLM runs with `seed=0`, so run-to-run variance is zero.
> Establishing that control was necessary; without it a 4.6-point gap is uninterpretable.
>
> ### Why: each MTP layer is a full attention layer with its own KV cache
>
> | | KV memory | KV tokens | per token | sessions @262 k |
> |---|---|---|---|---|
> | 1 head | 66.53 GiB | 1,877,478 | 38.1 KiB | 7.16× |
> | 2 heads | 65.67 GiB | 1,750,067 | **40.4 KiB** | 6.68× |
>
> +6 % KV per token, consistent with adding one more KV-bearing layer to the target's 16.
>
> And it explains the acceptance drop: with **one** reused head, its KV accumulates *every* draft
> position. With two heads, layer 1 only sees the positions it serves — a **gappy KV history**.
> Identical weights, different state, different predictions.
>
> ### Two premises corrected
>
> 1. **Gate 0's criterion was wrong.** "Two identical heads must behave like one reused head" is
>    false *by construction*, not because of a loading bug. The wiring is fine — it loads, drafts
>    3.00/step, and indexes correctly. My expectation was the defective part.
> 2. **Extra heads are not nearly free.** The original claim — "`k` heads read once each, same bytes
>    as one head read `k` times" — holds for *weights* but ignores KV. Each head adds ~6 % KV per
>    token, so four heads cost roughly 5.5× rather than 7.16× full-context sessions. Milder than
>    DSpark's 7.18 → 1.93, but the same category of cost I held against it.
>
> ### What this does to the case
>
> A copied head starts **below** the reused head at its own position (72.1 % vs 76.7 %). Training
> must therefore first recover 4.6 points from the gappy-KV handicap before delivering any gain.
> The upside range from §3 still stands *if* specialisation works, but the starting point is worse
> and the KV cost eats part of the win.
>
> **Revised Gate 1 (below): a *trained* head at position 1 must exceed 76.7 %** — the reused head's
> number, not the copied head's 72.1 %. Beating the copy is not an achievement; beating today is.

**Target:** `unsloth/Qwen3.8-27B-NVFP4` on GB10 · **Written:** 2026-08-17

---

## 1. The idea in one paragraph

Qwen3.8 ships **one** MTP head, reused at every speculative position. Its acceptance decays
**87.2 → 76.7 → 67.2 %** across positions 0–2, and `nspec=4` collapses outright. The head consumes
hidden states derived from its own previous guesses, so it is being asked to work on an input
distribution it was never fitted to — a **distribution shift that compounds with depth**.

Give each position its own copy of the head, fine-tuned on the inputs that position actually sees.
Fine-tuning is the right tool for distribution shift, and the shared 87 %-capability is already
trained — there is nothing to relearn.

**Why this is not the DSpark plan again:** that one died because an external drafter must pay its
own bandwidth *and* out-predict a head that ships inside the target. Extra MTP heads are cheaper,
though **not free** (Phase 0 corrected this): `k` heads are read **once each**, exactly like one
head read `k` times, so weight bytes per step are unchanged — but each head is a full attention
layer and adds **~6 % KV per token**. Measured head size is **745 MB**, not the 0.25 GB first
assumed.

---

## 2. What is already proven, and what is assumed

**Proven by inspection — the engine side needs no work:**

```python
# vllm/model_executor/models/qwen3_5_mtp.py
self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)   # :80
    for idx in range(self.num_mtp_layers)                            # :122  builds N layers
current_step_idx = spec_step_idx % self.num_mtp_layers               # :162  indexes them
```

vLLM already builds and indexes multiple MTP layers. Our checkpoint carries 15 tensors under
`mtp.layers.0.*` plus `mtp.fc`, with `text_config.mtp_num_hidden_layers: 1`. Producing
`mtp.layers.1.*` / `mtp.layers.2.*` and bumping that number is the whole integration — no vLLM
patch, no new architecture name, no `--trust-remote-code` payload. Every one of those was a
blocker for DSpark.

**Not proven — this is the question the plan exists to answer:** how much of the 87 → 77 → 67 decay
is *drift* (fixable by specialising a head per position) versus *intrinsic difficulty* (token t+2
is simply harder to predict, and no architecture fixes that). One hint points each way: MTP's decay
factor (~0.88/position) is the gentlest of the three drafters we measured, which argues drift is
not dominant; but `nspec=4` **collapsing** rather than degrading gracefully is exactly the signature
of compounding error.

**Phase 1 answers this with one head.** That is the whole reason the plan is sequential.

---

## 3. The bar

Measured on 8 real agent contexts, acceptance from
`vllm:spec_decode_num_accepted_tokens_per_pos_total`:

| position | current (one reused head) | a specialised head must beat |
|---|---|---|
| 0 | 87.2 % | — (unchanged, it is the same head) |
| 1 | 76.7 % | **> 76.7 %** |
| 2 | 67.2 % | **> 67.2 %** |
| 3 | collapses | anything usable is new capability |

Today: **3.31 tokens/step at 24.8 GB = 7.50 GB per generated token** (22.6 GB target + 3 reads of
the 745 MB head).

Payoff range at `k=4`, using the **measured** 745 MB per head:

| scenario | tokens/step | GB/token | throughput |
|---|---|---|---|
| no decay (87/87/87/87) | 4.49 | 5.70 | **+32 %** |
| gentle decay (87/85/82/79) | 4.33 | 5.91 | +27 % |
| today's decay, no compounding | 4.10 | 6.24 | +20 % |
| pessimistic (87/77/67/58) | 3.89 | 6.58 | **+14 %** |

Even the pessimistic case wins on throughput, because the fourth position costs one extra 745 MB
read and contributes ~0.58 accepted tokens. That asymmetry is the argument — but weigh it against
the KV cost: four heads take full-context concurrency from 7.16× to roughly **5.5×**. On an agent
fleet running long contexts, that trade is not obviously worth +14 %; on single-stream latency it
is.

---

## 4. Phases

### Phase 0 — DONE. Multi-layer MTP loads; identity does not hold
**Cost:** ~1 h · **Result: see the box at the top**

Duplicated `mtp.layers.0.*` to `.layers.1.*` unchanged, `mtp_num_hidden_layers: 2`, served, measured.
Loading and indexing work. Behaviour is **not** identical, because each layer carries its own KV
cache and the second one sees a gappy history.

**Gate 0 — passed on the part that mattered** (weights load under the expected names, `% num_mtp_layers`
indexes as the code reads), **failed on my criterion**, which assumed an identity that the
architecture does not provide. Cost of finding out: one hour, no training.

> Worth keeping as a lesson: the plumbing test was right to run, but its expected answer was wrong.
> A gate whose expected value you have not derived from the architecture only tests your assumption.

### Phase 1 — Fine-tune ONE head, for position 1 only
**Cost:** ~6–10 h · **This is the experiment; everything after is repetition**

1. Extract training pairs: run target + frozen head 0 over our agent traces, capture
   (hidden state produced at position 1, true token t+2). This is the same offline feature
   extraction the DSpark plan specified — the tooling is written.
2. Initialise head 1 from head 0's weights (BF16 copy — see A4 on quantization).
3. Fine-tune only head 1. Target frozen, head 0 frozen.
4. Serve with `mtp_num_hidden_layers: 2`, measure per-position acceptance on the same 8 contexts.

**Gate 1:** position-1 acceptance **> 80 %**, measured against the **reused head's 76.7 %** — not
against the copied head's 72.1 %. Training must first recover the 4.6-point gappy-KV handicap and
then beat today; only the second part is a gain. Below 76.7 % the decay is intrinsic difficulty
rather than drift, extra heads cannot deliver the projected range, and the plan stops with a
genuinely useful answer.

### Phase 2 — Head 2, then k=4
**Cost:** ~6–10 h per head

Repeat for position 2 (bar: > 67.2 %). Then add head 3 and test `nspec=4` — today impossible.
Re-measure throughput end-to-end with `/opt/llm/replay_bench.py` on the 24-context set.

**Gate 2:** end-to-end throughput on real agent prompts **> 26.4 tok/s median** (today's MTP n=3).
Acceptance improvements that do not show up here do not count.

### Phase 3 — Quantize and ship
Quantize the new heads to match the checkpoint, re-verify acceptance did not regress, then flip
`mtp_num_hidden_layers` in the served config. Keep the single-head config as the rollback.

---

## 5. Assumptions

| # | assumption | confidence | how to verify | if wrong |
|---|---|---|---|---|
| A1 | The decay is substantially drift, not intrinsic difficulty | **low** | Phase 1 | payoff collapses toward +0 %; plan stops |
| A2 | ~~vLLM's `% num_mtp_layers` indexing works as read~~ | **CONFIRMED** | Phase 0 | — |
| A8 | Extra heads cost only weights, not KV | **REFUTED** | Phase 0 | ~6 % KV/token per head; 4 heads ≈ 5.5× not 7.16× sessions |
| A9 | A head trained for its position can overcome the gappy-KV handicap | **low** | Gate 1 | plan stops |
| A3 | A fine-tuned copy can specialise to one position | medium | Phase 1 | same as A1 in effect |
| A4 | We can train in BF16 and re-quantize to NVFP4 without losing the gain | **low** | Phase 3 | gain may not survive quantization — measure before shipping |
| A5 | Our 1.34 M assistant tokens suffice to *specialise* (not train) a 0.4 B head | medium | Gate 1 | generate more traces (~12 instances/h) |
| A6 | No public tooling — SpecForge lists EAGLE3/DFlash/Domino/DSpark/P-EAGLE, **not MTP** | **high (it is absent)** | read the docs | custom training loop needed; see §7 |
| A7 | Extra heads do not disturb the hybrid attention / KV layout | medium | Phase 0 | KV accounting changes; re-measure sizing |

**A1 is the plan.** Everything else is engineering around it.

---

## 6. Cost

| phase | wall-clock |
|---|---|
| 0 — plumbing | ~1 h |
| 1 — one head, the real experiment | ~6–10 h |
| 2 — two more heads + k=4 | ~12–20 h |
| 3 — quantize, verify, ship | ~3 h |
| **to the decisive answer (Gate 1)** | **~1 day** |
| full path | ~3 days |

Per-head compute scales from the DSpark estimate: 0.4 B versus 1.36 B parameters, fine-tune rather
than from scratch, same corpus.

---

## 7. The one real gap: training code

No public trainer covers Qwen MTP heads. But the loop is small, and smaller than it looks:

- the target is **frozen** and only needs forward passes — no optimizer state for 27 B
- the trainable part is **one transformer layer** (~0.4 B), which fits comfortably
- the objective is ordinary next-token prediction with teacher forcing from the target
- the feature extraction is the same offline path the DSpark plan already specified

This is a training script, not a framework port. It is still the largest single piece of work here,
and it should be written **after** Gate 0, not before.

---

## 8. Explicitly rejected, with the measurements

| option | result |
|---|---|
| `RadixArk/Qwen3.8-27B-DSpark`, k=7 | 2.73 tokens/step, **−27 %** vs MTP on real agent traffic |
| DSpark k=14 | worse than k=7 |
| n-gram k=4 / k=8 | 2.26 / 2.62 tokens/step — position-0 copyability only ~42–45 % |
| `suffix` | not run; searches the same generated text, so the same ceiling applies |
| quantizing the DSpark drafter | halves the bandwidth penalty (2.21× → 1.46×) but cannot close a **prediction** gap |
| training a drafter *against the MTP head* | caps the student at the teacher, and MTP has no signal past position 2 |
| 4-bit KV (`turboquant_*`) | rejected on quality grounds, 2026-08-17 |

---

## 9. First step

Phase 0. Copy the head unchanged, set the count to 2, serve, and check the numbers do **not** move.
An hour, no training, and it either clears the road or ends the plan.
