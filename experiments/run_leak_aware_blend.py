#!/usr/bin/env python3
"""Extend iso_gapdrop to appearance_ce and evaluate 0.5 blends vs published 7.43.

Reuses ordprog_full iso_gapdrop from leaktrain_v2; trains appearance_ce under
erm + iso_gapdrop (same recipe). Reports seed-averaged-prediction MAE on LG-10.

Usage:
  PY=/root/anaconda3/envs/apds/bin/python
  $PY run_leak_aware_blend.py all
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TAB = OUT / "tables"
RELEASE = (
    ROOT
    / "elsarticle/release_tables"
)
sys.path.insert(0, str(ROOT))

from training.ordprog_net import VARIANTS, OrdProgNet  # noqa: E402

SEEDS = [0, 1, 2]
MAX_EPOCHS = 16
FOLD = "multi_gap_guard10"
IMAGENET_CKPT = Path("/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
EXPERT = "appearance_ce"
METHODS = {
    "erm": dict(iso=False, gapdrop_late=False, softmix=False),
    "iso_gapdrop": dict(iso=True, iso_alpha=1.0, gapdrop_late=True, drop_widths=(2, 3), softmix=False),
}


def stage_macro_mae(y, p):
    return float(pd.DataFrame({"g": y, "e": np.abs(y - p)}).groupby("g").e.mean().mean())


def load_pass_arrays():
    a = np.load(OUT / "sampled_data.npz")
    return a["stages"], a["keys"]


def isolation_weights(stages: np.ndarray) -> dict:
    uniq = np.array(sorted(set(int(s) for s in stages)))
    dmin = {}
    for s in uniq:
        others = uniq[uniq != s]
        dmin[int(s)] = float(np.min(np.abs(others - s))) if len(others) else 1.0
    vals = np.array(list(dmin.values()), dtype=np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    span = max(hi - lo, 1e-6)
    return {s: (d - lo) / span for s, d in dmin.items()}


def _train_jobs(job_list, worker_id: int):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.set_device(worker_id)
    device = torch.device(f"cuda:{worker_id}")

    protocol = json.loads((OUT / "protocol.json").read_text())
    fold = next(f for f in protocol["folds"] if f["name"] == FOLD)
    g, keys = load_pass_arrays()
    images = (
        torch.from_numpy(np.load(OUT / "sampled_images.npy").copy())
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device)
    )
    gp = np.repeat(g, 3)
    targets = torch.tensor(gp / 169.0, dtype=torch.float32, device=device)
    stage_t = torch.tensor(gp, device=device, dtype=torch.long)
    c3 = torch.tensor(np.where(gp <= 40, 0, np.where(gp <= 110, 1, 2)), device=device, dtype=torch.long)
    c5 = torch.tensor(np.minimum((gp - 1) // 34, 4), device=device, dtype=torch.long)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    weight = torch.load(IMAGENET_CKPT, map_location="cpu", weights_only=True)
    cfg = VARIANTS[EXPERT]

    def seed_all(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def normalize(ids, augment=False):
        x = images[ids].float().div_(255.0)
        if augment:
            flip = torch.rand(len(ids), device=device) < 0.5
            x[flip] = x[flip].flip(-1)
        return x

    def make_net(seed: int):
        seed_all(seed)
        model = OrdProgNet(cfg, imagenet_state=weight).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        return model, opt, sch

    def gapdrop_keep(train_stages, widths):
        stages = np.array(sorted(set(int(s) for s in train_stages)))
        if len(stages) < 8:
            return stages
        w = int(random.choice(list(widths)))
        w = min(w, max(1, len(stages) // 4))
        c = int(random.choice(stages.tolist()))
        keep = stages[(stages < c - w) | (stages > c + w)]
        if len(keep) < max(6, len(stages) // 4):
            return stages
        return keep

    def element_loss(o3, o5, op, y3, y5, yp):
        ce3 = F.cross_entropy(o3, y3, reduction="none") if cfg.ce3_weight else 0.0
        pr = (op - yp).pow(2) if cfg.progress_loss != "smooth_l1" else F.smooth_l1_loss(op, yp, reduction="none")
        if cfg.use_coral:
            targets_c = torch.stack([(y5 > k).float() for k in range(4)], dim=1)
            coral = F.binary_cross_entropy_with_logits(o5, targets_c, reduction="none").mean(1)
            return cfg.ce3_weight * ce3 + cfg.progress_weight * pr + cfg.coral_weight * coral
        ce5 = F.cross_entropy(o5, y5, reduction="none")
        return cfg.ce3_weight * ce3 + cfg.progress_weight * pr + cfg.ce5_weight * ce5

    def train_epoch(model, opt, ids, *, mcfg, train_stage_values, epoch, iso_map):
        model.train()
        use_ids = ids
        if mcfg.get("gapdrop_late") and epoch > MAX_EPOCHS // 2:
            keep_stages = set(gapdrop_keep(train_stage_values, mcfg["drop_widths"]).tolist())
            keep = torch.tensor(
                [int(stage_t[i].item()) in keep_stages for i in ids.tolist()],
                device=device,
                dtype=torch.bool,
            )
            if int(keep.sum()) >= 64:
                use_ids = ids[keep]
        order = use_ids[torch.randperm(len(use_ids), device=device)]
        total = 0.0
        n_seen = 0
        alpha = float(mcfg.get("iso_alpha", 1.0))
        do_iso = bool(mcfg.get("iso"))
        for batch in order.split(64):
            x = normalize(batch, True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o3, o5, op = model(x, mean, std)
                per = element_loss(o3, o5, op, c3[batch], c5[batch], targets[batch])
                if do_iso:
                    w = torch.tensor(
                        [1.0 + alpha * iso_map.get(int(stage_t[i].item()), 0.0) for i in batch.tolist()],
                        device=device,
                        dtype=per.dtype,
                    )
                    loss = (per * w).mean()
                else:
                    loss = per.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
            n_seen += len(batch)
        return total / max(n_seen, 1)

    @torch.inference_mode()
    def predict_pass(model, pass_mask):
        ids = torch.tensor(np.flatnonzero(np.repeat(pass_mask, 3)), device=device)
        model.eval()
        parts = []
        for batch in ids.split(96):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                op = model(normalize(batch), mean, std)[2]
            parts.append(op.float().cpu().numpy())
        return np.concatenate(parts).reshape(-1, 3).mean(1) * 169.0

    my_jobs = [j for i, j in enumerate(job_list) if i % 2 == worker_id]
    print(f"BLEND_CE_WORKER{worker_id}_JOBS", len(my_jobs), "/", len(job_list), flush=True)
    for method, seed, tag in my_jobs:
        result_path = OUT / f"{tag}.json"
        if result_path.exists() and (OUT / f"{tag}_predictions.csv").exists():
            print("SKIP", tag, flush=True)
            continue
        mcfg = METHODS[method]
        fit = np.isin(g, fold["inner_fit_stages"])
        val = np.isin(g, fold["inner_val_stages"])
        train = fit | val
        test = np.isin(g, fold["test_stages"])
        core = np.isin(g, fold["core_test_stages"])
        fitids = torch.tensor(np.flatnonzero(np.repeat(fit, 3)), device=device)
        trainids = torch.tensor(np.flatnonzero(np.repeat(train, 3)), device=device)
        fit_stages, train_stages = g[fit], g[train]
        iso_fit, iso_train = isolation_weights(fit_stages), isolation_weights(train_stages)

        start = time.time()
        model, opt, sch = make_net(seed)
        best = (float("inf"), None)
        for epoch in range(1, MAX_EPOCHS + 1):
            train_epoch(model, opt, fitids, mcfg=mcfg, train_stage_values=fit_stages, epoch=epoch, iso_map=iso_fit)
            sch.step()
            vm = stage_macro_mae(g[val], predict_pass(model, val))
            if vm < best[0]:
                best = (vm, epoch)
        del model, opt, sch
        torch.cuda.empty_cache()

        model, opt, sch = make_net(seed)
        for epoch in range(1, best[1] + 1):
            train_epoch(
                model, opt, trainids, mcfg=mcfg, train_stage_values=train_stages, epoch=epoch, iso_map=iso_train
            )
            sch.step()
        ph = predict_pass(model, test)
        core_m = core[test]
        ckpt = OUT / f"{tag}.pt"
        torch.save(
            {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "method": method, "seed": seed, "epochs": best[1], "expert": EXPERT},
            ckpt,
        )
        pd.DataFrame(
            {
                "fold": FOLD,
                "method": method,
                "model": EXPERT,
                "seed": seed,
                "stage": g[test].astype(int),
                "pass_id": keys[test, 1].astype(int),
                "prediction": ph,
                "in_core45": core[test],
            }
        ).to_csv(OUT / f"{tag}_predictions.csv", index=False)
        row = dict(
            fold=FOLD,
            method=method,
            model=EXPERT,
            seed=seed,
            selected_epochs=best[1],
            inner_val_mae=best[0],
            mae_stage_macro_core45=stage_macro_mae(g[test][core_m], ph[core_m]),
            elapsed_seconds=time.time() - start,
            checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        )
        result_path.write_text(json.dumps(row, indent=2))
        print("COMPLETE", worker_id, tag, "core45", round(row["mae_stage_macro_core45"], 3), flush=True)
        del model, opt, sch
        torch.cuda.empty_cache()


def train_jobs():
    return [
        (m, s, f"leaktrain_v2__{FOLD}__{EXPERT}__{m}__seed{s}")
        for m in METHODS
        for s in SEEDS
    ]


def train_main():
    jobs = train_jobs()
    missing = [
        j
        for j in jobs
        if not ((OUT / f"{j[2]}.json").exists() and (OUT / f"{j[2]}_predictions.csv").exists())
    ]
    print("MISSING_CE_JOBS", len(missing), "/", len(jobs), flush=True)
    if not missing:
        return
    procs = [
        subprocess.Popen([sys.executable, str(Path(__file__)), "train", "worker", str(wid)], cwd=str(OUT))
        for wid in (0, 1)
    ]
    rc = [p.wait() for p in procs]
    if any(rc):
        raise SystemExit(f"worker failed: {rc}")


def train_worker(wid: int):
    _train_jobs(train_jobs(), wid)


def _load_core_preds(path_pattern: str):
    preds, stages = [], None
    for seed in SEEDS:
        p = pd.read_csv(path_pattern.format(seed=seed))
        p = p[p["in_core45"]].sort_values(["stage", "pass_id"])
        preds.append(p["prediction"].values)
        stages = p["stage"].values
    return stages, np.stack(preds, 0)


def summarize():
    TAB.mkdir(exist_ok=True)
    # singles
    singles = {}
    # published ERM
    stages, coral_pub = _load_core_preds(str(OUT / f"{FOLD}__ordprog_full__seed{{seed}}_predictions.csv"))
    _, ce_pub = _load_core_preds(str(OUT / f"{FOLD}__appearance_ce__seed{{seed}}_predictions.csv"))
    singles["coral_erm_pub"] = stage_macro_mae(stages, coral_pub.mean(0))
    singles["ce_erm_pub"] = stage_macro_mae(stages, ce_pub.mean(0))
    singles["blend_pub_0.5"] = stage_macro_mae(stages, (0.5 * (coral_pub + ce_pub)).mean(0))

    # v2 coral iso (already trained)
    _, coral_iso = _load_core_preds(
        str(OUT / f"leaktrain_v2__{FOLD}__ordprog_full__iso_gapdrop__seed{{seed}}_predictions.csv")
    )
    singles["coral_iso"] = stage_macro_mae(stages, coral_iso.mean(0))

    # newly trained ce
    _, ce_erm = _load_core_preds(
        str(OUT / f"leaktrain_v2__{FOLD}__appearance_ce__erm__seed{{seed}}_predictions.csv")
    )
    _, ce_iso = _load_core_preds(
        str(OUT / f"leaktrain_v2__{FOLD}__appearance_ce__iso_gapdrop__seed{{seed}}_predictions.csv")
    )
    singles["ce_erm_retrain"] = stage_macro_mae(stages, ce_erm.mean(0))
    singles["ce_iso"] = stage_macro_mae(stages, ce_iso.mean(0))

    blends = {
        "blend_pub_coral_erm+ce_erm": stage_macro_mae(stages, (0.5 * (coral_pub + ce_pub)).mean(0)),
        "blend_coral_iso+ce_pub": stage_macro_mae(stages, (0.5 * (coral_iso + ce_pub)).mean(0)),
        "blend_coral_iso+ce_erm_retrain": stage_macro_mae(stages, (0.5 * (coral_iso + ce_erm)).mean(0)),
        "blend_coral_iso+ce_iso": stage_macro_mae(stages, (0.5 * (coral_iso + ce_iso)).mean(0)),
        "blend_coral_pub+ce_iso": stage_macro_mae(stages, (0.5 * (coral_pub + ce_iso)).mean(0)),
    }

    target = 7.4311
    board = []
    for k, v in {**singles, **blends}.items():
        board.append(dict(config=k, mae=v, beats_7_43=bool(v < target - 1e-6), delta_vs_7_43=v - target))
    board_df = pd.DataFrame(board).sort_values("mae")
    best_blend = min(blends.items(), key=lambda kv: kv[1])
    headline = dict(
        published_blend_target=target,
        coral_iso_alone=singles["coral_iso"],
        ce_iso_alone=singles["ce_iso"],
        best_blend_name=best_blend[0],
        best_blend_mae=best_blend[1],
        beats_published_blend=bool(best_blend[1] < target - 1e-6),
        delta_best_blend_minus_7_43=best_blend[1] - target,
        note="0.5 equal blend of seed-averaged progress predictions on LG-10 core45",
    )
    board_df.to_csv(TAB / "leak_aware_blend_board.csv", index=False)
    board_df.to_csv(RELEASE / "leak_aware_blend_board.csv", index=False)
    (TAB / "leak_aware_blend_headline.json").write_text(json.dumps(headline, indent=2))
    (RELEASE / "leak_aware_blend_headline.json").write_text(json.dumps(headline, indent=2))
    print(board_df.to_string(index=False), flush=True)
    print(json.dumps(headline, indent=2), flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "train" and len(sys.argv) >= 4 and sys.argv[2] == "worker":
        train_worker(int(sys.argv[3]))
        return
    if cmd == "train":
        train_main()
        return
    if cmd == "summarize":
        summarize()
        return
    if cmd == "all":
        train_main()
        summarize()
        return
    raise SystemExit(cmd)


if __name__ == "__main__":
    main()
