"""Decision-oriented (triplet) verification head vs the regression baseline.

Train: ResNet-18 + L2 embedding, triplet loss with negatives restricted to
|delta stage| <= 10.  Test: margin between the claimed stage prototype and the
best neighbouring prototype, calibrated to the same false-review rate as the
baseline (0.265).
"""
import os
from pathlib import Path
ROOT_DIR = Path(os.environ.get("SOFTADV_ROOT", ".")).resolve()
import csv
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm
import torchvision.transforms as T

ROOT = str(ROOT_DIR) + "/"
FRAMES_CSV = f"{ROOT}/outputs/processed/frames_with_passes.csv"
CORE_CSV = "aord_erm_s0.csv"          # supplies the 234 core (stage, pass) pairs
OUT_JSON = "triplet_result.json"
H, W = 128, 768
STAGES = 169
EPOCHS = 8
P, K = 8, 4
LR = 3e-4
MARGIN = 0.3
SEED = 0

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = True

tf = T.Compose([T.Resize((H, W)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def load_index():
    frames = []
    with open(FRAMES_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            frames.append({"path": r["path"], "stage": int(r["grind"]),
                           "pass": int(r["pass_id"])})
    core = set()
    with open(CORE_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("in_core45", "False").strip().lower() == "true":
                core.add((int(r["stage"]), int(r["pass_id"])))
    return frames, core

class FrameSet(Dataset):
    def __init__(self, items):
        self.by_path = {it["path"]: it for it in items}
        self.keys = list(self.by_path)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, key):
        it = self.by_path[key]
        try:
            img = Image.open(it["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (W, H))
        return tf(img), it["stage"]

class PKBatchSampler(torch.utils.data.Sampler):
    """P stages x K frames per batch."""

    def __init__(self, stages_to_frames, p, k, steps):
        self.map = stages_to_frames
        self.stages = [s for s, v in stages_to_frames.items() if len(v) >= 2]
        self.p, self.k, self.steps = p, k, steps

    def __iter__(self):
        for _ in range(self.steps):
            ys = random.sample(self.stages, self.p)
            batch = []
            for y in ys:
                pool = self.map[y]
                batch.extend(random.sample(pool, min(self.k, len(pool))))
            yield batch

    def __len__(self):
        return self.steps

class Embedder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        self.proj = nn.Linear(512, dim)

    def forward(self, x):
        z = self.backbone(x).flatten(1)
        return F.normalize(self.proj(z), dim=1)

def batch_hard_triplet(emb, ys):
    d = 1.0 - emb @ emb.t()
    n = len(ys)
    y = torch.tensor(ys, device=emb.device)
    gap = (y[:, None] - y[None, :]).abs()
    same = gap == 0
    near = (gap >= 1) & (gap <= 10)
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    same = same & ~eye

    pos_d = d.masked_fill(~same, -1.0).max(dim=1).values
    neg_d = d.masked_fill(~near, 1e9).min(dim=1).values
    valid = same.any(dim=1) & near.any(dim=1)
    loss = F.relu(MARGIN + pos_d[valid] - neg_d[valid])
    return loss.mean() if loss.numel() else emb.sum() * 0

def train(train_frames):
    by_stage = {}
    for f in train_frames:
        by_stage.setdefault(f["stage"], []).append(f["path"])
    ds = FrameSet(train_frames)
    lookup = {f["path"]: f["stage"] for f in train_frames}
    steps = math.ceil(len(train_frames) / (P * K))
    sampler = PKBatchSampler(by_stage, P, K, steps)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=6, pin_memory=True)

    model = Embedder().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps)

    for ep in range(EPOCHS):
        model.train()
        total, seen = 0.0, 0
        for step, (xb, ys) in enumerate(loader):
            xb = xb.cuda(non_blocking=True)
            emb = model(xb)
            loss = batch_hard_triplet(emb, list(ys))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * xb.size(0)
            seen += xb.size(0)
            if step % 100 == 0:
                print(f"ep{ep} step{step}/{steps} loss {loss.item():.4f}", flush=True)
        print(f"epoch {ep} mean loss {total / max(1, seen):.4f}", flush=True)
    return model

@torch.no_grad()
def embed_paths(model, paths, limit=8):
    model.eval()
    out = []
    if len(paths) > limit:
        idx = np.linspace(0, len(paths) - 1, limit).astype(int)
        paths = [paths[i] for i in idx]
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        x = tf(img).unsqueeze(0).cuda()
        out.append(model(x).cpu())
    if not out:
        return None
    return torch.cat(out).mean(0)

def main():
    frames, core = load_index()
    print(f"frames {len(frames)}, core pairs {len(core)}", flush=True)
    train_frames = [f for f in frames if (f["stage"], f["pass"]) not in core]
    test_frames = [f for f in frames if (f["stage"], f["pass"]) in core]
    print(f"train frames {len(train_frames)}, test frames {len(test_frames)}", flush=True)

    model = train(train_frames)
    torch.save(model.state_dict(), "triplet_embedder.pt")

    # prototypes from training frames
    by_stage = {}
    for f in train_frames:
        by_stage.setdefault(f["stage"], []).append(f["path"])
    proto = {}
    for s in range(1, STAGES + 1):
        paths = by_stage.get(s)
        if not paths:
            continue
        e = embed_paths(model, paths)
        if e is not None:
            proto[s] = F.normalize(e, dim=0)
    print(f"prototypes {len(proto)}", flush=True)

    # pass embeddings for the core set
    by_pass = {}
    for f in test_frames:
        by_pass.setdefault((f["stage"], f["pass"]), []).append(f["path"])
    emb = {}
    for key, paths in by_pass.items():
        e = embed_paths(model, paths)
        if e is not None:
            emb[key] = F.normalize(e, dim=0)
    print(f"pass embeddings {len(emb)}", flush=True)

    def margin(e, claim):
        if claim not in proto:
            return None
        s = float(e @ proto[claim])
        best = -2.0
        for d in range(-10, 11):
            if d == 0:
                continue
            g = claim + d
            if g in proto:
                best = max(best, float(e @ proto[g]))
        return s - best if best > -2 else 0.0

    keys = sorted(emb)
    stages = sorted({k[0] for k in keys})
    rng = random.Random(0)
    rng.shuffle(stages)
    cal_stages = set(stages[: len(stages) // 2])

    def score_set(keys_subset):
        vals = []
        for k in keys_subset:
            m = margin(emb[k], k[0])
            if m is not None:
                vals.append((k, m))
        return vals

    cal = score_set([k for k in keys if k[0] in cal_stages])
    test = score_set([k for k in keys if k[0] not in cal_stages])
    if len(cal) < 20 or len(test) < 20:
        print("not enough scored passes", len(cal), len(test))
        return

    q = sorted(m for _, m in cal)[int(0.265 * len(cal))]
    fa = sum(1 for _, m in test if m < q) / len(test)

    def det(lo, hi, draws=20):
        r = random.Random(7)
        hit = tot = 0
        for k, _ in test:
            g, _p = k
            for _ in range(draws):
                d = r.randint(lo, hi)
                claim = g + d
                if claim < 1 or claim > STAGES:
                    continue
                m = margin(emb[k], claim)
                if m is None:
                    continue
                tot += 1
                hit += m < q
        return hit / max(1, tot)

    res = {"q": q, "n_cal": len(cal), "n_test": len(test), "FA": fa,
           "Det_small": det(3, 8), "Det_large": det(15, 40)}
    print(json.dumps(res, indent=2), flush=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(res, fh, indent=2)

if __name__ == "__main__":
    main()
