import torch, glob
from types import SimpleNamespace
from safetensors import safe_open
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
dequant = DFlashQwen3Model._dequant_kv_slice

FP8 = "/home/jschmied/dflash2-tcclaviger-fp8/model.safetensors"
BF16 = glob.glob("/opt/llm/models/qwen38-27b-dflash2/*.safetensors")

def get(files, names):
    out={}
    for f in ([files] if isinstance(files,str) else files):
        with safe_open(f, framework="pt") as h:
            for k in names:
                if k in h.keys(): out[k]=h.get_tensor(k)
    return out

L = "layers.0.self_attn"
w = get(FP8, [f"{L}.{p}_proj.weight" for p in "qkv"] +
             [f"{L}.{p}_proj.weight_scale_inv" for p in "qkv"])
ref = get(BF16, [f"{L}.{p}_proj.weight" for p in "qkv"])

# So fusioniert vLLM: Zeilen von q, k, v aneinander -- Gewicht wie Skala
weight = torch.cat([w[f"{L}.{p}_proj.weight"] for p in "qkv"], dim=0)
scale  = torch.cat([w[f"{L}.{p}_proj.weight_scale_inv"] for p in "qkv"], dim=0)
q_size = ref[f"{L}.q_proj.weight"].shape[0]
print(f"  fusioniert: weight {tuple(weight.shape)} {weight.dtype}, "
      f"scale_inv {tuple(scale.shape)}, q_size {q_size}")

attn = SimpleNamespace(
    qkv_proj=SimpleNamespace(weight=weight, weight_scale_inv=scale,
                             weight_block_size=[128,128], input_size=weight.shape[1]),
    q_size=q_size)

got = dequant(attn, torch.bfloat16)
want = torch.cat([ref[f"{L}.k_proj.weight"], ref[f"{L}.v_proj.weight"]], dim=0).to(torch.bfloat16)
print(f"  Ergebnis {tuple(got.shape)} vs BF16-Original {tuple(want.shape)}")
assert got.shape == want.shape

d = (got.float()-want.float()).abs()
rel = d / want.float().abs().clamp_min(1e-6)
print(f"\n  median rel. Fehler: {rel.median():.4%}")
print(f"  p99   rel. Fehler: {rel.flatten().float().quantile(0.99):.4%}")
a=got.double().flatten(); b=want.double().flatten()
print(f"  Kosinus (float64): {(a@b/(a.norm()*b.norm())).item():.6f}")
print(f"  Kosinus (float32, wie zuvor): {torch.nn.functional.cosine_similarity(got.float().flatten(), want.float().flatten(), dim=0).item():.6f}  <- >1 = Akkumulationsfehler")

# Gegenprobe: waere der Scale ein KEHRWERT, muesste die Division passen
inv = weight[q_size:].float() / scale[q_size//128:].float().repeat_interleave(128,0).repeat_interleave(128,1)[:got.shape[0],:got.shape[1]]
rel_inv = ((inv-want.float()).abs()/want.float().abs().clamp_min(1e-6))
print(f"\n  haette man DIVIDIERT: median rel. Fehler {rel_inv.median():.2%}"
      f"  -> {'ebenfalls plausibel?!' if rel_inv.median()<0.1 else 'grob falsch, Multiplikation ist richtig'}")
