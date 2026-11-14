"""Linear probe on the triplet embedding, evaluated on held-out stages.

The prototype-based verifier cannot score held-out stages (no prototypes exist
for them), so this evaluates the representation instead: ridge regression from
embedding to stage, then the same threshold rule.
"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import csv
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

H, W = 128, 768
ROOT = str(ROOT_DIR) + "/"
tf = T.Compose([T.Resize((H, W)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

from run_triplet import Embedder  # reuse the architecture

frames = []
with open(f"{ROOT}/outputs/processed/frames_with_passes.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        frames.append({"path": r["path"], "stage": int(r["grind"]), "pass": int(r["pass_id"])})

core = set()
with open("aord_erm_s0.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r.get("in_core45", "False").strip().lower() == "true":
            core.add((int(r["stage"]), int(r["pass_id"])))

by_pass = {}
for f in frames:
    by_pass.setdefault((f["stage"], f["pass"]), []).append(f["path"])

train_keys = [k for k in by_pass if k not in core]
test_keys = [k for k in by_pass if k in core]
print(f"train passes {len(train_keys)}, test passes {len(test_keys)}", flush=True)

model = Embedder().cuda()
model.load_state_dict(torch.load("triplet_embedder.pt"))
model.eval()

@torch.no_grad()
def embed_pass(paths, limit=8):
    if len(paths) > limit:
        idx = np.linspace(0, len(paths) - 1, limit).astype(int)
        paths = [paths[i] for i in idx]
    xs = []
    for p in paths:
        try:
            xs.append(tf(Image.open(p).convert("RGB")))
        except Exception:
            pass
    if not xs:
        return None
    xb = torch.stack(xs).cuda()
    return model(xb).mean(0).cpu()

def embed_all(keys):
    out = {}
    for i, k in enumerate(keys):
        e = embed_pass(by_pass[k])
        if e is not None:
            out[k] = e
        if i % 100 == 0:
            print(f"  {i}/{len(keys)}", flush=True)
    return out

print("embedding train passes...", flush=True)
Etr = embed_all(train_keys)
print("embedding test passes...", flush=True)
Ete = embed_all(test_keys)

Xtr = torch.stack([Etr[k] for k in Etr]).numpy()
ytr = np.array([k[0] for k in Etr], dtype=np.float32)
Xte = torch.stack([Ete[k] for k in Ete]).numpy()
yte = np.array([k[0] for k in Ete], dtype=np.float32)

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Xtr = (Xtr - mu) / sd
Xte = (Xte - mu) / sd
Xtr1 = np.hstack([Xtr, np.ones((len(Xtr), 1))])
w = np.linalg.solve(Xtr1.T @ Xtr1 + 1e-2 * np.eye(Xtr1.shape[1]), Xtr1.T @ ytr)
pred = np.hstack([Xte, np.ones((len(Xte), 1))]) @ w

mae = np.abs(pred - yte).mean()
print(f"held-out stage MAE {mae:.3f}", flush=True)

keys = list(Ete)
err = np.abs(pred - yte)

stages = sorted({k[0] for k in keys})
random.Random(0).shuffle(stages)
cal = set(stages[: len(stages) // 2])
mask_cal = np.array([k[0] in cal for k in keys])
mask_te = ~mask_cal
q = np.percentile(err[mask_cal], 73.5) if mask_cal.sum() else 10.0
fa = float((err[mask_te] > q).mean())

def det(lo, hi, qq, draws=20):
    r = random.Random(7)
    hit = tot = 0
    for i, k in enumerate(keys):
        if not mask_te[i]:
            continue
        for _ in range(draws):
            d = r.randint(lo, hi)
            claim = yte[i] + d
            hit += abs(pred[i] - claim) > qq
            tot += 1
    return hit / max(1, tot)

q2 = float(np.percentile(err[mask_te], 73.5))
fa2 = float((err[mask_te] > q2).mean())
res = {"mae_heldout": float(mae),
       "heldout_cal": {"q": float(q), "FA": fa,
                       "Det_small": det(3, 8, q), "Det_large": det(15, 40, q)},
       "matched_FA": {"q": q2, "FA": fa2,
                      "Det_small": det(3, 8, q2), "Det_large": det(15, 40, q2)}}
print(json.dumps(res, indent=2), flush=True)
json.dump(res, open("linear_result.json", "w"), indent=2)
