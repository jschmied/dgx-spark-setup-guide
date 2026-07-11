# Appendix: per-phase temperature for agentic coding (Pi + the local NVFP4 endpoint)

The fleet endpoint (Appendix A) is one OpenAI-compatible URL on `:8080`. A single global
sampling temperature is a **compromise** for an agentic coding workflow, because the two things
an agent does want *opposite* settings:

| phase | wants | temperature |
|---|---|---|
| **planning / reasoning** | explore alternatives, recover when the first idea is wrong | **higher** (~0.6–0.8) |
| **execution** (tool calls, edits, structured output) | determinism, valid tool-args, reproducibility | **low** (~0.2) |

This page wires that split into **[Pi](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/README.md)**
(the terminal coding agent) against the local endpoint, using its per-agent model config and
[`pi-subagents`](https://pi.dev/packages/pi-subagents) — **not** a per-turn temperature knob, for a
reason that matters on this model (below).

## Why the split is worth it (and its cost)

Measured on this box's `qwen36-35b-a3b` (see Appendix B):

- **Tool-call reliability rises sharply as temp falls** — 1/8 valid at temp 1.0 (the model's
  `generation_config` default!), 6/8 at 0.6, **7/8 at 0.2**. Low temp is the reliability setting.
- **Low temp also *speeds up* decode** — a colder draft agrees with the target more often, so the
  **MTP spec-decode acceptance rises** (we saw 0.57→0.76 swings), and decode t/s scales with it. So
  for execution, low temp is a rare quality-*and*-speed win.
- **The cost of low temp is exploration.** At temp 0 our SWE-bench runs *looped to the 250-call step
  limit* on hard tasks — with no randomness, a stuck agent can't back out and try another path. That
  recovery/exploration is exactly what the **planner** phase needs, so give it the higher temp.

The split *recovers* the exploration you'd lose by isolating high temp to the phase that benefits
(planning) while keeping execution cold and reliable.

## Why per-*agent*, not per-*turn*

Qwen3.6 is a **reasoning model**: every worker turn *internally* both reasons (wants exploration)
and emits the tool call (wants precision), in one generation. You **cannot** set different
temperatures for the thinking vs. the tool-args within a single request. So the only honest seam is
the **agent boundary** — a planner agent (hot) that produces a plan, and a worker agent (cold) that
executes it. This is also Pi's grain: its recommended flow is **clarify → planner → worker → fresh
reviewers → worker**, with the planner writing a plan to a file the worker reads.

## Setup

> Pi sets sampling at the **model/provider config** level, so per-phase temperature = **two model
> entries** pointing at the *same* endpoint with *different* `temperature`, assigned to different
> subagents. The exact config keys below are illustrative — **verify against Pi's own docs**
> ([setup guide](https://www.bitdoze.com/pi-coding-agent-setup-guide/)); the *structure* is what matters.

**1. An OpenAI-compatible provider → the local endpoint.** Point Pi at `:8080` with one of the
router/vLLM API keys (`/etc/llama-server/api_keys.txt`):

```
provider "local-gb10":
  type: openai            # OpenAI-compatible
  base_url: http://127.0.0.1:8080/v1
  api_key: sk-...         # any key from /etc/llama-server/api_keys.txt
```

**2. Two model entries — same served model, different temperature:**

```
model "qwen-planner":    # HOT — exploration
  provider: local-gb10
  model: qwen36-35b-a3b
  temperature: 0.7

model "qwen-worker":     # COLD — reliable execution
  provider: local-gb10
  model: qwen36-35b-a3b
  temperature: 0.2
```

**3. Assign them to subagents** — planner/reviewer use `qwen-planner`, worker/executor use
`qwen-worker` (in each agent's frontmatter `model:`, or via the agent tool's `model` override when a
parent delegates).

## The gotcha: the server has a *default* temperature

The vLLM backend pins a **server-side default** via `--override-generation-config` (currently
`temperature 0.6`, from Appendix A/B). That default only applies **when the client sends no
temperature**. So for the hot/cold split to take effect, **Pi must actually send `temperature` in
each request** — if an agent sends nothing, it silently inherits the server's 0.6 and your worker is
*not* at 0.2.

**Verify it's really sending:** watch the vLLM log for the incoming sampling params, or curl the
endpoint as each agent would and confirm the request carries `temperature`. A quick check:

```bash
KEY=$(grep -vE '^\s*([#;]|$)' /etc/llama-server/api_keys.txt | head -1)
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-35b-a3b","messages":[{"role":"user","content":"ping"}],"temperature":0.2,"max_tokens":8}' \
  >/dev/null && echo "endpoint honors per-request temperature"
```

If Pi has no way to pass `temperature` through, the fallback is to **set the server default to your
execution value** (0.2) and only raise temp for the planner — but that inverts the safety (a
misconfigured client then defaults to the *cold* setting, which is the safer failure).

## Recommended temperatures

| agent role | temperature | rationale |
|---|---|---|
| planner / reviewer | **0.6–0.8** | exploration, alternative approaches, loop recovery |
| worker / executor | **0.2** | tool-call reliability (7/8), determinism, higher MTP acceptance |
| never | **1.0** (the model default) | 1/8 tool-call reliability — always override |

If you see workers **looping or getting stuck**, that's the low temp surfacing — nudge the *worker*
to ~0.3–0.4 (keeps most reliability, restores enough exploration to break loops), or lean harder on
the planner to pre-decompose the task so the worker never has to improvise.

## See also

- Appendix A — the vLLM/router `:8080` endpoint and API keys these agents point at.
- Appendix B — where the temperature↔tool-reliability and temperature↔MTP-acceptance numbers come
  from, and the `--override-generation-config` server default.
