# NVFP4 periodic table — sources and data

Backing material for the published map of `lm_head` × body quantization combinations on
Qwen3.8-27B / DGX Spark (GB10).

**The procedure for adding a model is the `nvfp4-table` skill**
(`~/.claude/skills/nvfp4-table/SKILL.md`). This directory is what it operates on.

| path | what |
| --- | --- |
| `nvfp4-periodensystem.html` | source of the published map |
| `acceptance-decay.html` | source of the acceptance-decay diagram |
| `data/cells.csv` | every measured cell, machine-readable — regenerate from the HTML |
| `data/divergence-ref.json` | **teacher-forced BF16 reference. Effectively irreplaceable** |
| `data/divergence-bf16.json` | BF16 top-1 baseline the Δtop-1 figures are measured against |
| `data/<run>/` | raw per-request JSON and vLLM `.metrics` per measurement run |
| `tools/measure-cell.sh` | three-arm template: MTP, divergence, DFlash2 |
| `tools/divergence.py` | teacher-forced logprob divergence |
| `tools/heartbeat.sh` | unconditional 20-minute status monitor for detached runs |
| `tools/toolcall-temp.sh` | tool-calling reliability across temperatures |

Regenerating `divergence-ref.json` requires serving BF16 Qwen3.8-27B, which took this box into a
global OOM at util 0.50 and was abandoned. Treat it as archival.

`/tmp` on this box is emptied on boot. The tools also live at `/opt/llm/divergence.py` and
`/opt/llm/divergence-ref.json`, but `/opt/llm` is not versioned — this directory is the store.

Findings are written up in `../../appendix/qwen38-27b-nvfp4-vllm027.md`, sections 7f–8.
