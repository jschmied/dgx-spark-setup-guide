#!/usr/bin/env python3
# DFlash2-Drafter nach FP8 quantisieren.
#
# Why on THIS box: we are bandwidth-bound, not memory-bound. Measured 231.8 GB/s
# of a 273 GB/s peak, and the drafter accounts for 15.5 % of the bytes read per
# step (it runs once per step -- block diffusion). Halving its linear weights
# removes ~7 % of the bytes, so ~+7.5 % throughput on paper, FOR AS LONG AS
# acceptance length holds (4.32 at the time of writing).
#
# NICHT quantisiert werden:
#   - the two-tap convolutions (attention_conv/mlp_conv): they carry the draft
#     across the block, which is the heart of the method
#   - the candidate_selector and its codebooks: it picks the path
#   - every norm
# 0.39 GB together -- little to save there, and not little to risk.
#
# W8A8 dynamic rather than W8A16: only the WEIGHT bytes matter for bandwidth,
# and the fp8 path is better worn in than the humming path.
import json, os, re, shutil, glob
import torch
from safetensors.torch import load_file, save_file

Q = "/opt/llm/models/qwen38-27b-dflash2"
Z = "/opt/llm/models/qwen38-27b-dflash2-fp8"
FP8_MAX = 448.0

ZIEL = re.compile(r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj|^fc)\.weight$")

os.makedirs(Z, exist_ok=True)
SHARDS = sorted(os.path.basename(f) for f in glob.glob(f"{Q}/*.safetensors"))
assert SHARDS, "keine safetensors gefunden"
neu_map = {}
q_anz = b_alt = b_new = 0

for sh in SHARDS:
    t = load_file(f"{Q}/{sh}")
    out = {}
    for k, v in t.items():
        b_alt += v.numel() * v.element_size()
        if ZIEL.search(k) and v.dim() == 2:
            w = v.to(torch.float32)
            amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = amax / FP8_MAX
            qw = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            out[k] = qw
            out[k.replace(".weight", ".weight_scale")] = scale.squeeze(-1).to(torch.float32)
            q_anz += 1
        else:
            out[k] = v
    for k, v in out.items():
        b_new += v.numel() * v.element_size()
        neu_map[k] = sh
    save_file(out, f"{Z}/{sh}", metadata={"format": "pt"})
    print(f"  {sh}: {len(t)} -> {len(out)} tensors")

if len(SHARDS) > 1:
    json.dump({"metadata": {}, "weight_map": neu_map},
              open(f"{Z}/model.safetensors.index.json", "w"), indent=1)

c = json.load(open(f"{Q}/config.json"))
c["quantization_config"] = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["re:.*self_attn\\.(q|k|v|o)_proj$",
                        "re:.*mlp\\.(gate|up|down)_proj$",
                        "re:^fc$"],
            "weights": {"num_bits": 8, "type": "float", "strategy": "channel",
                        "symmetric": True, "dynamic": False,
                        "observer": "memoryless_minmax"},
            "input_activations": {"num_bits": 8, "type": "float",
                                  "strategy": "token", "symmetric": True,
                                  "dynamic": True, "observer": None},
            "format": "float-quantized",
        }
    },
    "ignore": ["re:.*norm.*", "re:.*_conv.*", "re:.*candidate_selector.*",
               "re:.*mask_embedding.*"],
}
json.dump(c, open(f"{Z}/config.json", "w"), indent=1)

for f in os.listdir(Q):
    if f.endswith((".json", ".jinja", ".txt", ".md")) and f != "config.json":
        shutil.copy(f"{Q}/{f}", f"{Z}/{f}")

print(f"\n  {q_anz} Gewichte quantisiert")
print(f"  vorher {b_alt/1e9:.2f} GB  ->  nachher {b_new/1e9:.2f} GB  ({(b_new-b_alt)/b_alt*100:+.1f} %)")
print(f"  bytes per step: 21.0 + {b_new/1e9:.2f} = {21+b_new/1e9:.2f} GB (24.85 before)")

