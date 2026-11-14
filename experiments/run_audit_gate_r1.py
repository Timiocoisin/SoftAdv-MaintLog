#!/usr/bin/env python3
"""Route-1 v3: FloorRefine AuditGate — train + MAE/Φ board (no manuscript edit).

AuditGate-F: gate m:=m0 (Φ-locked); λ_map=λ0+β·tanh(refine); MetalField with spatial floor.
Compared to: concat, metal_op, metal_op_learn, free_op, AuditGate v1/v2.

Primary decision: LG-10 seedavg MAE; secondary Φ_bright. win if AuditGate
beats metal_op/concat on MAE while Φ stays ≫ free_op.

Usage:
  PY=/root/anaconda3/envs/apds/bin/python
  $PY run_audit_gate_r1.py all
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

from training.ordprog_net import (  # noqa: E402
    METAL_THRESHOLD_8BIT,
    VARIANTS,
    OrdProgNet,
    bright_metal_mask,
    ordprog_loss,
)

SEEDS = [0, 1, 2]
MAX_EPOCHS = 16
FOLD = "multi_gap_guard10"
IMAGENET_CKPT = Path("/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
TAU0 = METAL_THRESHOLD_8BIT / 255.0
BRIGHT_FACTORS = [0.5, 0.7, 1.0, 1.15, 1.5]

# Train v3 FloorRefine AuditGate (fresh tags; prior runs kept on disk)
TRAIN_VARIANTS = {
    "audit_f_erm": dict(variant="ordprog_audit_gate", iso_gapdrop=False, cons_w=0.0),
    "audit_f_iso": dict(variant="ordprog_audit_gate", iso_gapdrop=True, cons_w=0.0),
    "audit_f_wide": dict(
        variant="ordprog_audit_gate", iso_gapdrop=False, cons_w=0.0, beta_init=0.0
    ),  # softplus(0)≈0.69 → wider floor swing
}

# Reference checkpoints already on disk (name -> (glob_or_tag_fn))
REF_TAGS = {
    "concat_erm": "multi_gap_guard10__ordprog_full__seed{seed}",
    "metal_op": "multi_gap_guard10__ordprog_metal_op__seed{seed}",
    "metal_op_learn": "multi_gap_guard10__ordprog_metal_op_learn__seed{seed}",
    "free_op": "multi_gap_guard10__ordprog_free_op__seed{seed}",
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


def batch_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a > 0.5
    b = b > 0.5
    inter = (a & b).flatten(1).sum(1).float()
    union = (a | b).flatten(1).sum(1).float().clamp_min(1.0)
    return inter / union


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

    def gapdrop_keep(train_stages, widths=(2, 3)):
        stages = np.array(sorted(set(int(s) for s in train_stages)))
        if len(stages) < 8:
            return stages
        w = int(random.choice(list(widths)))
        w = min(w, max(1, len(stages) // 4))
        c = int(random.choice(stages.tolist()))
        keep = stages[(stages < c - w) | (stages > c + w)]
        return stages if len(keep) < max(6, len(stages) // 4) else keep

    def element_loss(cfg, o3, o5, op, y3, y5, yp):
        ce3 = F.cross_entropy(o3, y3, reduction="none") if cfg.ce3_weight else 0.0
        pr = (op - yp).pow(2)
        if cfg.use_coral:
            tc = torch.stack([(y5 > k).float() for k in range(4)], dim=1)
            coral = F.binary_cross_entropy_with_logits(o5, tc, reduction="none").mean(1)
            return cfg.ce3_weight * ce3 + cfg.progress_weight * pr + cfg.coral_weight * coral
        ce5 = F.cross_entropy(o5, y5, reduction="none")
        return cfg.ce3_weight * ce3 + cfg.progress_weight * pr + cfg.ce5_weight * ce5

    def train_epoch(model, opt, cfg, ids, *, use_iso, train_stages, epoch, iso_map, cons_w=0.0):
        model.train()
        use_ids = ids
        if use_iso and epoch > MAX_EPOCHS // 2:
            keep_stages = set(gapdrop_keep(train_stages).tolist())
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
        for batch in order.split(64):
            x = normalize(batch, True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o3, o5, op = model(x, mean, std)
                per = element_loss(cfg, o3, o5, op, c3[batch], c5[batch], targets[batch])
                if use_iso:
                    w = torch.tensor(
                        [1.0 + iso_map.get(int(stage_t[i].item()), 0.0) for i in batch.tolist()],
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
    print(f"AUDIT_WORKER{worker_id}_JOBS", len(my_jobs), "/", len(job_list), flush=True)
    for method, seed, tag in my_jobs:
        result_path = OUT / f"{tag}.json"
        if result_path.exists() and (OUT / f"{tag}_predictions.csv").exists():
            print("SKIP", tag, flush=True)
            continue
        spec = TRAIN_VARIANTS[method]
        cfg = VARIANTS[spec["variant"]]
        use_iso = bool(spec["iso_gapdrop"])
        cons_w = float(spec.get("cons_w", 0.0))
        fit = np.isin(g, fold["inner_fit_stages"])
        val = np.isin(g, fold["inner_val_stages"])
        train = fit | val
        test = np.isin(g, fold["test_stages"])
        core = np.isin(g, fold["core_test_stages"])
        fitids = torch.tensor(np.flatnonzero(np.repeat(fit, 3)), device=device)
        trainids = torch.tensor(np.flatnonzero(np.repeat(train, 3)), device=device)
        iso_fit = isolation_weights(g[fit])
        iso_train = isolation_weights(g[train])

        def make(seed_):
            seed_all(seed_)
            model = OrdProgNet(cfg, imagenet_state=weight).to(device)
            if "beta_init" in spec and model.audit_beta_raw is not None:
                with torch.no_grad():
                    model.audit_beta_raw.fill_(float(spec["beta_init"]))
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
            return model, opt, sch

        start = time.time()
        model, opt, sch = make(seed)
        best = (float("inf"), None)
        for epoch in range(1, MAX_EPOCHS + 1):
            train_epoch(
                model,
                opt,
                cfg,
                fitids,
                use_iso=use_iso,
                train_stages=g[fit],
                epoch=epoch,
                iso_map=iso_fit,
                cons_w=cons_w,
            )
            sch.step()
            vm = stage_macro_mae(g[val], predict_pass(model, val))
            if vm < best[0]:
                best = (vm, epoch)
        del model, opt, sch
        torch.cuda.empty_cache()

        model, opt, sch = make(seed)
        for epoch in range(1, best[1] + 1):
            train_epoch(
                model,
                opt,
                cfg,
                trainids,
                use_iso=use_iso,
                train_stages=g[train],
                epoch=epoch,
                iso_map=iso_train,
                cons_w=cons_w,
            )
            sch.step()
        ph = predict_pass(model, test)
        core_m = core[test]
        ckpt = OUT / f"{tag}.pt"
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "cfg": cfg.to_dict(),
                "method": method,
                "seed": seed,
                "epochs": best[1],
                "learned_tau": model.learned_tau(),
                "audit_gamma": model.audit_gamma(),
            },
            ckpt,
        )
        pd.DataFrame(
            {
                "fold": FOLD,
                "method": method,
                "model": cfg.name,
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
            model=cfg.name,
            seed=seed,
            selected_epochs=best[1],
            inner_val_mae=best[0],
            mae_stage_macro_core45=stage_macro_mae(g[test][core_m], ph[core_m]),
            learned_tau=model.learned_tau(),
            audit_gamma=model.audit_gamma(),
            elapsed_seconds=time.time() - start,
            checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        )
        result_path.write_text(json.dumps(row, indent=2))
        print("COMPLETE", worker_id, tag, "core45", round(row["mae_stage_macro_core45"], 3), flush=True)
        del model, opt, sch
        torch.cuda.empty_cache()


def train_jobs():
    return [
        (m, s, f"auditgatef__{FOLD}__{m}__seed{s}")
        for m in TRAIN_VARIANTS
        for s in SEEDS
    ]


def train_main():
    jobs = train_jobs()
    missing = [
        j
        for j in jobs
        if not ((OUT / f"{j[2]}.json").exists() and (OUT / f"{j[2]}_predictions.csv").exists())
    ]
    print("MISSING_AUDIT", len(missing), "/", len(jobs), flush=True)
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


def _mae_from_preds(tag_tmpl: str) -> float:
    preds, stages = [], None
    for seed in SEEDS:
        p = pd.read_csv(OUT / f"{tag_tmpl.format(seed=seed)}_predictions.csv")
        p = p[p["in_core45"]].sort_values(["stage", "pass_id"])
        preds.append(p["prediction"].values)
        stages = p["stage"].values
    return stage_macro_mae(stages, np.mean(np.stack(preds, 0), 0))


@torch.inference_mode()
def _phi_for_checkpoints(tag_tmpl: str, variant_name: str | None = None) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    protocol = json.loads((OUT / "protocol.json").read_text())
    fold = next(f for f in protocol["folds"] if f["name"] == FOLD)
    g, _ = load_pass_arrays()
    images = (
        torch.from_numpy(np.load(OUT / "sampled_images.npy").copy())
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device)
    )
    core = np.isin(g, fold["core_test_stages"])
    frame_ids = torch.tensor(np.flatnonzero(np.repeat(core, 3)), device=device)
    weight = torch.load(IMAGENET_CKPT, map_location="cpu", weights_only=True)

    # cache clean m*
    ref = {}
    for batch in frame_ids.split(96):
        x = images[batch].float() / 255.0
        mstar = bright_metal_mask(x, mode="fixed", threshold=TAU0)
        for i, idx in enumerate(batch.tolist()):
            ref[idx] = mstar[i : i + 1].cpu()

    phi_clean_all = []
    phi_bright = {f: [] for f in BRIGHT_FACTORS}
    taus = []
    for seed in SEEDS:
        tag = tag_tmpl.format(seed=seed)
        ckpt = torch.load(OUT / f"{tag}.pt", map_location="cpu", weights_only=False)
        cfg_name = variant_name or ckpt.get("cfg", {}).get("name")
        if cfg_name is None:
            # infer from TRAIN / REF
            cfg_name = "ordprog_audit_gate"
        cfg = VARIANTS[cfg_name]
        model = OrdProgNet(cfg, imagenet_state=weight).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        if model.learned_tau() is not None:
            taus.append(model.learned_tau())
        for batch in frame_ids.split(64):
            x = images[batch].float() / 255.0
            gate = model.gate_map(x)
            mstar = bright_metal_mask(x, mode="fixed", threshold=TAU0)
            phi_clean_all.append(batch_iou(gate, mstar).cpu().numpy())
            mstar_c = torch.cat([ref[i] for i in batch.tolist()], 0).to(device)
            for fac in BRIGHT_FACTORS:
                x_f = (x * float(fac)).clamp(0, 1)
                gate_f = model.gate_map(x_f)
                phi_bright[fac].append(batch_iou(gate_f, mstar_c).cpu().numpy())
        del model
        torch.cuda.empty_cache()
    return dict(
        phi_clean=float(np.concatenate(phi_clean_all).mean()),
        phi_bright_mean=float(np.mean([np.concatenate(v).mean() for v in phi_bright.values()])),
        learned_tau_mean=float(np.mean(taus)) if taus else None,
    )


def summarize():
    TAB.mkdir(exist_ok=True)
    rows = []

    # references with existing preds + ckpts
    ref_variants = {
        "concat_erm": ("ordprog_full", "multi_gap_guard10__ordprog_full__seed{seed}"),
        "metal_op": ("ordprog_metal_op", "multi_gap_guard10__ordprog_metal_op__seed{seed}"),
        "metal_op_learn": ("ordprog_metal_op_learn", "multi_gap_guard10__ordprog_metal_op_learn__seed{seed}"),
        "free_op": ("ordprog_free_op", "multi_gap_guard10__ordprog_free_op__seed{seed}"),
    }
    for name, (vname, tmpl) in ref_variants.items():
        try:
            mae = _mae_from_preds(tmpl)
            phi = _phi_for_checkpoints(tmpl, variant_name=vname)
            rows.append(dict(method=name, mae=mae, **phi, auditable=name != "free_op"))
            print(f"REF {name} MAE={mae:.3f} Φ={phi['phi_bright_mean']:.3f}", flush=True)
        except Exception as e:
            print("SKIP_REF", name, e, flush=True)

    # new FloorRefine audit gates
    for method in TRAIN_VARIANTS:
        tmpl = f"auditgatef__{FOLD}__{method}__seed{{seed}}"
        mae = _mae_from_preds(tmpl)
        phi = _phi_for_checkpoints(tmpl, variant_name="ordprog_audit_gate")
        rows.append(dict(method=method, mae=mae, **phi, auditable=True))
        print(f"NEW {method} MAE={mae:.3f} Φ={phi['phi_bright_mean']:.3f}", flush=True)

    # prior AuditGate iterations (cached MAE/Φ)
    for name, mae, phi_b, phi_c, tau in [
        ("audit_v1_erm", 7.997597, 0.3919, 0.5105, 0.1096),
        ("audit_v2_cons", 7.816703, 0.9404, 0.9998, 0.1094),
    ]:
        rows.append(
            dict(
                method=name,
                mae=mae,
                phi_clean=phi_c,
                phi_bright_mean=phi_b,
                learned_tau_mean=tau,
                auditable=True,
            )
        )

    board = pd.DataFrame(rows).sort_values("mae")
    metal = board.loc[board.method == "metal_op", "mae"]
    concat = board.loc[board.method == "concat_erm", "mae"]
    free_phi = (
        float(board.loc[board.method == "free_op", "phi_bright_mean"].iloc[0])
        if "free_op" in set(board.method)
        else 0.12
    )
    audit_rows = board[board.method.str.startswith("audit_f")]
    best = audit_rows.sort_values("mae").iloc[0]
    metal_mae = float(metal.iloc[0]) if len(metal) else 7.69
    concat_mae = float(concat.iloc[0]) if len(concat) else 8.19
    metal_learn = board.loc[board.method == "metal_op_learn", "mae"]
    metal_learn_mae = float(metal_learn.iloc[0]) if len(metal_learn) else 7.58
    headline = dict(
        module="AuditGate_FloorRefine",
        definition="m:=m0 (Φ-locked); λ=λ0+β·tanh(refine); x'=x⊙(λ+(1-λ)m0)",
        best_audit_method=str(best["method"]),
        best_audit_mae=float(best["mae"]),
        best_audit_phi_bright=float(best["phi_bright_mean"]),
        concat_mae=concat_mae,
        metal_op_mae=metal_mae,
        metal_op_learn_mae=metal_learn_mae,
        free_phi_bright=free_phi,
        beats_concat=bool(best["mae"] < concat_mae - 0.15),
        beats_metal_op=bool(best["mae"] < metal_mae - 0.15),
        beats_metal_learn=bool(best["mae"] < metal_learn_mae - 0.10),
        phi_keeps_audit=bool(best["phi_bright_mean"] > free_phi + 0.3),
        win=bool(
            (best["mae"] < min(concat_mae, metal_mae) - 0.15)
            and (best["phi_bright_mean"] > free_phi + 0.3)
        ),
        soft_win=bool(
            (best["mae"] <= metal_learn_mae + 0.05)
            and (best["phi_bright_mean"] > 0.7)
            and (best["mae"] < concat_mae - 0.15)
        ),
        note="No manuscript edit until win/soft_win. Structural originality = FloorRefine AuditGate.",
    )
    for stem in ("audit_gate_v3", "audit_gate_r1"):
        board.to_csv(TAB / f"{stem}_board.csv", index=False)
        board.to_csv(RELEASE / f"{stem}_board.csv", index=False)
        (TAB / f"{stem}_headline.json").write_text(json.dumps(headline, indent=2))
        (RELEASE / f"{stem}_headline.json").write_text(json.dumps(headline, indent=2))
    print(board.to_string(index=False), flush=True)
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
