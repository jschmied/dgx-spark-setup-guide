#!/usr/bin/env python3
# DFlash2-Drafter nach FP8 quantisieren.
#
# Warum auf DIESER Box: wir sind bandbreitengebunden, nicht speichergebunden.
# Gemessen 231.8 GB/s von 273 GB/s Spitze, und der Drafter macht 15.5 % der
# Bytes je Schritt aus (er laeuft einmal je Schritt, Block-Diffusion).
# Halbiert man seine Linear-Gewichte, fallen ~7 % der Bytes weg -> rechnerisch
# +7.5 % Durchsatz, SOLANGE die Akzeptanzlaenge haelt (heute 4.32).
#
# NICHT quantisiert werden:
#   - die Zwei-Tap-Faltungen (attention_conv/mlp_conv): sie halten den Entwurf
#     ueber die Blocklaenge, das ist der Kern des Verfahrens
#   - der candidate_selector samt Codebuechern: er waehlt den Pfad
#   - alle Normen
# Zusammen 0.39 GB -- das Sparpotenzial dort ist klein, das Risiko nicht.
#
# W8A8 dynamisch statt W8A16: fuer die Bandbreite zaehlen nur die
# GEWICHTSbytes, und der fp8-Pfad ist besser eingefahren als der humming-Pfad.
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
q_anz = b_alt = b_neu = 0

for sh in SHARDS:
    t = load_file(f"{Q}/{sh}")
    aus = {}
    for k, v in t.items():
        b_alt += v.numel() * v.element_size()
        if ZIEL.search(k) and v.dim() == 2:
            w = v.to(torch.float32)
            amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale = amax / FP8_MAX
            qw = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            aus[k] = qw
            aus[k.replace(".weight", ".weight_scale")] = scale.squeeze(-1).to(torch.float32)
            q_anz += 1
        else:
            aus[k] = v
    for k, v in aus.items():
        b_neu += v.numel() * v.element_size()
        neu_map[k] = sh
    save_file(aus, f"{Z}/{sh}", metadata={"format": "pt"})
    print(f"  {sh}: {len(t)} -> {len(aus)} Tensoren")

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
print(f"  vorher {b_alt/1e9:.2f} GB  ->  nachher {b_neu/1e9:.2f} GB  ({(b_neu-b_alt)/b_alt*100:+.1f} %)")
print(f"  Bytes je Schritt: 21.0 + {b_neu/1e9:.2f} = {21+b_neu/1e9:.2f} GB (heute 24.85)")

