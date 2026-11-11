#!/usr/bin/env python3
"""Corruption + spatial transfer for Audit-OrdNet (Route A), no manuscript edit.

Corruption: evaluate existing multi_gap_guard10 / auditord / auditgatef checkpoints
under protocol_extras corruption families (seedavg progress → stage-macro MAE on core45).

Spatial: retrain on spatial_A_to_B / spatial_B_to_A for:
  - aord (audit_ord_coral, wide β)
  - metal_op_learn
  - audit_f (ordprog_audit_gate, wide β)
and compare to published Table3 OrdProg/baseline numbers.

Usage:
  PY=/root/anaconda3/envs/apds/bin/python
  $PY run_audit_ord_extras.py all
  $PY run_audit_ord_extras.py corrupt
  $PY run_audit_ord_extras.py spatial
  $PY run_audit_ord_extras.py summarize
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
    / "IET_Image_Processing_LaTeX_Template/WileyDesign/Optimal-Design-layout/release_tables"
)
sys.path.insert(0, str(ROOT))

from training.ordprog_net import (  # noqa: E402
    METAL_THRESHOLD_8BIT,
    VARIANTS,
    build_model,
    decode_predictions,
    ordprog_loss,
)

# Reuse corruption helper from P0 extras
import run_p0_extras as extras  # noqa: E402

SEEDS = [0, 1, 2]
MAX_EPOCHS = 16
FOLD = "multi_gap_guard10"
IMAGENET_CKPT = Path("/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
BATCH_CORRUPT = 48
BATCH_SPATIAL = 48

# Methods evaluated under corruption (guard-10 checkpoints)
CORRUPT_METHODS = {
    "aord_erm": dict(
        variant="audit_ord_coral",
        tag="auditord__multi_gap_guard10__aord_erm__seed{seed}",
        beta_init=0.0,
    ),
    "aord_narrow": dict(
        variant="audit_ord_coral",
        tag="auditord__multi_gap_guard10__aord_narrow__seed{seed}",
        beta_init=-2.0,
    ),
    "audit_f_wide": dict(
        variant="ordprog_audit_gate",
        tag="auditgatef__multi_gap_guard10__audit_f_wide__seed{seed}",
        beta_init=0.0,
    ),
    "metal_op_learn": dict(
        variant="ordprog_metal_op_learn",
        tag="multi_gap_guard10__ordprog_metal_op_learn__seed{seed}",
    ),
    "metal_op": dict(
        variant="ordprog_metal_op",
        tag="multi_gap_guard10__ordprog_metal_op__seed{seed}",
    ),
    "concat_erm": dict(
        variant="ordprog_full",
        tag="multi_gap_guard10__ordprog_full__seed{seed}",
    ),
    "free_op": dict(
        variant="ordprog_free_op",
        tag="multi_gap_guard10__ordprog_free_op__seed{seed}",
    ),
    "baseline_ce": dict(
        variant="baseline_ce",
        tag="multi_gap_guard10__baseline_ce__seed{seed}",
    ),
}

# Spatial retrains
SPATIAL_METHODS = {
    "aord_erm": dict(variant="audit_ord_coral", beta_init=0.0),
    "aord_rob": dict(variant="audit_ord_coral", beta_init=0.0, aug_p=0.65),
    "aord_aug": dict(variant="audit_ord_coral", beta_init=0.0, aug_p=0.45),
    "metal_op_learn": dict(variant="ordprog_metal_op_learn"),
    "audit_f_wide": dict(variant="ordprog_audit_gate", beta_init=0.0),
}


def stage_macro_mae(y, p):
    return float(pd.DataFrame({"g": y, "e": np.abs(y - p)}).groupby("g").e.mean().mean())


def spearman(y, p):
    return float(pd.Series(y).corr(pd.Series(p), method="spearman")) if len(y) > 1 else float("nan")


def classification_acc(y, p):
    return float((y == p).mean()) if len(y) else float("nan")


def load_pass_arrays():
    a = np.load(OUT / "sampled_data.npz")
    return a["stages"], a["keys"]


def _ensure_spatial_proto():
    if (OUT / "protocol_extras.json").exists() and (OUT / "pass_spatial.csv").exists():
        return json.loads((OUT / "protocol_extras.json").read_text())
    return extras.prepare()[0]


def _load_ckpt_model(spec: dict, seed: int, device, weight):
    cfg = VARIANTS[spec["variant"]]
    model = build_model(cfg, imagenet_state=weight).to(device)
    tag = spec["tag"].format(seed=seed)
    ckpt = torch.load(OUT / f"{tag}.pt", map_location="cpu", weights_only=False)
    # FloorRefine / AuditOrd may need beta fill if missing keys; strict=False for safety
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print("WARN_MISSING_KEYS", tag, missing[:5], flush=True)
    if "beta_init" in spec and hasattr(model, "audit_beta_raw") and model.audit_beta_raw is not None:
        # only fill if not in ckpt
        if any("audit_beta_raw" in k for k in missing):
            with torch.no_grad():
                model.audit_beta_raw.fill_(float(spec["beta_init"]))
    model.eval()
    return model, cfg


def run_corruption():
    device = torch.device("cuda:0")
    proto = _ensure_spatial_proto()
    mg = json.loads((OUT / "protocol.json").read_text())
    fold = next(f for f in mg["folds"] if f["name"] == FOLD)
    g, _ = load_pass_arrays()
    images = (
        torch.from_numpy(np.load(OUT / "sampled_images.npy").copy())
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device)
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    weight = torch.load(IMAGENET_CKPT, map_location="cpu", weights_only=True)
    test = np.isin(g, fold["test_stages"])
    core = np.isin(g, fold["core_test_stages"])
    core_m = core[test]
    y = g[test]

    # Drop methods whose checkpoints are missing
    usable = {}
    for name, spec in CORRUPT_METHODS.items():
        ok = all((OUT / f"{spec['tag'].format(seed=s)}.pt").exists() for s in SEEDS)
        if ok:
            usable[name] = spec
        else:
            print("SKIP_CORRUPT_METHOD", name, "missing ckpt", flush=True)

    models = {
        name: [_load_ckpt_model(spec, s, device, weight) for s in SEEDS]
        for name, spec in usable.items()
    }

    @torch.inference_mode()
    def predict(name, family, severity):
        ids = torch.tensor(np.flatnonzero(np.repeat(test, 3)), device=device)
        outs = []
        for model, cfg in models[name]:
            parts = []
            for batch in ids.split(BATCH_CORRUPT):
                x = images[batch].float() / 255.0
                if family != "clean":
                    x = extras._corrupt_batch(x, family, severity, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    op = model(x, mean, std)[2]
                parts.append(op.float().cpu().numpy())
            outs.append(np.concatenate(parts).reshape(-1, 3).mean(1) * 169.0)
        return np.mean(np.stack(outs, 0), 0)

    families = {"clean": [None]}
    families.update(proto["corruption"]["families"])
    rows = []
    for family, sevs in families.items():
        for sev in sevs:
            print("CORRUPT", family, sev, flush=True)
            for name in usable:
                p = predict(name, family, sev)
                rows.append(
                    dict(
                        checkpoint_fold=FOLD,
                        corruption=family,
                        severity="" if sev is None else sev,
                        method=name,
                        mae_stage_macro_core45=stage_macro_mae(y[core_m], p[core_m]),
                        mae_pass_core45=float(np.abs(p[core_m] - y[core_m]).mean()),
                    )
                )
    tab = pd.DataFrame(rows)
    TAB.mkdir(exist_ok=True)
    tab.to_csv(TAB / "audit_ord_corrupt_raw.csv", index=False)
    tab.to_csv(RELEASE / "audit_ord_corrupt_raw.csv", index=False)

    summ = (
        tab[tab.corruption != "clean"]
        .groupby(["method", "corruption"])["mae_stage_macro_core45"]
        .mean()
        .unstack()
    )
    clean = tab[tab.corruption == "clean"].set_index("method")["mae_stage_macro_core45"]
    summ.insert(0, "clean", clean)
    summ["mean_corrupt"] = summ.drop(columns=["clean"]).mean(axis=1)
    summ = summ.sort_values("mean_corrupt")
    summ.to_csv(TAB / "audit_ord_corrupt_summary.csv")
    summ.to_csv(RELEASE / "audit_ord_corrupt_summary.csv")
    print(summ.round(3).to_string(), flush=True)
    return tab, summ


def spatial_jobs():
    proto = _ensure_spatial_proto()
    jobs = []
    for fold in proto["spatial_folds"]:
        for method in SPATIAL_METHODS:
            for seed in SEEDS:
                tag = f"spatial_{fold['name']}__{method}__seed{seed}"
                # fold name already starts with spatial_
                tag = f"{fold['name']}__{method}__seed{seed}"
                jobs.append((fold["name"], method, seed, tag))
    return jobs


def _spatial_train_worker(worker_id: int):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.set_device(worker_id)
    device = torch.device(f"cuda:{worker_id}")

    proto = _ensure_spatial_proto()
    pass_df = pd.read_csv(OUT / "pass_spatial.csv")
    g, keys = load_pass_arrays()
    images = (
        torch.from_numpy(np.load(OUT / "sampled_images.npy").copy())
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device)
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    weight = torch.load(IMAGENET_CKPT, map_location="cpu", weights_only=True)
    gp = np.repeat(g, 3)
    targets = torch.tensor(gp / 169.0, dtype=torch.float32, device=device)
    c3 = torch.tensor(np.where(gp <= 40, 0, np.where(gp <= 110, 1, 2)), device=device, dtype=torch.long)
    c5 = torch.tensor(np.minimum((gp - 1) // 34, 4), device=device, dtype=torch.long)

    fold_by_name = {f["name"]: f for f in proto["spatial_folds"]}

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    AUG_POOL = [
        ("blur_sigma", 0.5),
        ("blur_sigma", 1.0),
        ("jpeg_quality", 80),
        ("jpeg_quality", 50),
        ("gauss_noise_sigma", 0.0196078431372549),
        ("gauss_noise_sigma", 0.0392156862745098),
        ("center_crop_frac", 0.9),
        ("brightness", 0.85),
        ("brightness", 1.15),
    ]

    def batch_x(ids, augment=False, aug_p=0.0):
        x = images[ids].float().div_(255.0)
        if augment:
            flip = torch.rand(len(ids), device=device) < 0.5
            x[flip] = x[flip].flip(-1)
            if aug_p > 0:
                # batch-level corrupt (faster than per-sample)
                if random.random() < aug_p:
                    fam, sev = random.choice(AUG_POOL)
                    x = extras._corrupt_batch(x, fam, sev, device)
        return x

    def make_net(method, seed):
        seed_all(seed)
        spec = SPATIAL_METHODS[method]
        cfg = VARIANTS[spec["variant"]]
        model = build_model(cfg, imagenet_state=weight).to(device)
        if "beta_init" in spec and hasattr(model, "audit_beta_raw") and model.audit_beta_raw is not None:
            with torch.no_grad():
                model.audit_beta_raw.fill_(float(spec["beta_init"]))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        return model, opt, sch, cfg, float(spec.get("aug_p", 0.0))

    def train_epoch(model, opt, cfg, ids, aug_p=0.0):
        model.train()
        order = ids[torch.randperm(len(ids), device=device)]
        total = 0.0
        for batch in order.split(BATCH_SPATIAL):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o3, o5, op = model(batch_x(batch, True, aug_p=aug_p), mean, std)
                loss = ordprog_loss(o3, o5, op, c3[batch], c5[batch], targets[batch], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
        return total / max(len(ids), 1)

    @torch.inference_mode()
    def predict_pass(model, cfg, pass_indices):
        mask = np.zeros(len(g), dtype=bool)
        mask[np.asarray(pass_indices)] = True
        ids = torch.tensor(np.flatnonzero(np.repeat(mask, 3)), device=device)
        model.eval()
        parts, y3s, y5s = [], [], []
        for batch in ids.split(64):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o3, o5, op = model(batch_x(batch), mean, std)
            dec = decode_predictions(o3.float(), o5.float(), op.float(), cfg)
            parts.append(dec["progress"].cpu().numpy())
            y3s.append(dec["y3"].cpu().numpy())
            y5s.append(dec["y5"].cpu().numpy())
        prog = np.concatenate(parts).reshape(-1, 3).mean(1) * 169.0
        y3 = pd.DataFrame(np.concatenate(y3s).reshape(-1, 3)).mode(axis=1)[0].to_numpy().astype(int)
        y5 = pd.DataFrame(np.concatenate(y5s).reshape(-1, 3)).mode(axis=1)[0].to_numpy().astype(int)
        return prog, y3, y5, np.flatnonzero(mask)

    jobs = spatial_jobs()
    my = [j for i, j in enumerate(jobs) if i % 2 == worker_id]
    print(f"SPATIAL_WORKER{worker_id}_JOBS", len(my), "/", len(jobs), flush=True)
    for fold_name, method, seed, tag in my:
        result_path = OUT / f"{tag}.json"
        if result_path.exists() and (OUT / f"{tag}_predictions.csv").exists():
            print("SKIP_SPATIAL", tag, flush=True)
            continue
        fold = fold_by_name[fold_name]
        train_passes = np.array(fold["train_pass_indices"])
        test_passes = np.array(fold["test_pass_indices"])
        train_stages_of = pass_df.loc[train_passes, "stage"].to_numpy()
        fit_m = np.isin(train_stages_of, fold["inner_fit_stages"])
        val_m = np.isin(train_stages_of, fold["inner_val_stages"])
        fit_passes = train_passes[fit_m]
        val_passes = train_passes[val_m]
        fitids = torch.tensor(
            np.concatenate([np.arange(p * 3, p * 3 + 3) for p in fit_passes]), device=device
        )
        trainids = torch.tensor(
            np.concatenate([np.arange(p * 3, p * 3 + 3) for p in train_passes]), device=device
        )

        start = time.time()
        model, opt, sch, cfg, aug_p = make_net(method, seed)
        best = (float("inf"), None)
        for epoch in range(1, MAX_EPOCHS + 1):
            train_epoch(model, opt, cfg, fitids, aug_p=aug_p)
            sch.step()
            ph, _, _, val_idx = predict_pass(model, cfg, val_passes)
            vm = stage_macro_mae(g[val_idx], ph)
            if vm < best[0]:
                best = (vm, epoch)
            if epoch % 4 == 0:
                print("SELECT", worker_id, tag, epoch, round(vm, 3), "best", best[1], flush=True)
        del model, opt, sch
        torch.cuda.empty_cache()

        model, opt, sch, cfg, aug_p = make_net(method, seed)
        for epoch in range(1, best[1] + 1):
            train_epoch(model, opt, cfg, trainids, aug_p=aug_p)
            sch.step()
        ph, p3, p5, te_idx = predict_pass(model, cfg, test_passes)
        y3_true = np.where(g[te_idx] <= 40, 0, np.where(g[te_idx] <= 110, 1, 2))
        y5_true = np.minimum((g[te_idx] - 1) // 34, 4)

        ckpt = OUT / f"{tag}.pt"
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "fold": fold_name,
                "method": method,
                "variant": SPATIAL_METHODS[method]["variant"],
                "cfg": cfg.to_dict(),
                "seed": seed,
                "epochs": best[1],
                "learned_tau": model.learned_tau(),
                "audit_gamma": model.audit_gamma() if hasattr(model, "audit_gamma") else None,
            },
            ckpt,
        )
        pred = pd.DataFrame(
            {
                "fold": fold_name,
                "model": method,
                "seed": seed,
                "stage": g[te_idx],
                "pass_id": keys[te_idx, 1],
                "pass_index": te_idx,
                "prediction": ph,
                "pred_y3": p3,
                "pred_y5": p5,
                "error": np.abs(ph - g[te_idx]),
            }
        )
        pred.to_csv(OUT / f"{tag}_predictions.csv", index=False)
        row = dict(
            fold=fold_name,
            model=method,
            seed=seed,
            selected_epochs=best[1],
            inner_val_mae=best[0],
            mae_pass=float(np.abs(ph - g[te_idx]).mean()),
            mae_stage_macro=stage_macro_mae(g[te_idx], ph),
            spearman_pass=spearman(g[te_idx], ph),
            acc3_pass=classification_acc(y3_true, p3),
            acc5_pass=classification_acc(y5_true, p5),
            acc3_from_prog=classification_acc(
                y3_true, np.where(ph <= 40, 0, np.where(ph <= 110, 1, 2)).astype(int)
            ),
            elapsed_seconds=time.time() - start,
            checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        )
        result_path.write_text(json.dumps(row, indent=2))
        print(
            "COMPLETE_SPATIAL",
            worker_id,
            tag,
            "MAE",
            round(row["mae_stage_macro"], 3),
            flush=True,
        )
        del model, opt, sch
        torch.cuda.empty_cache()


def spatial_main():
    jobs = spatial_jobs()
    missing = [
        j
        for j in jobs
        if not ((OUT / f"{j[3]}.json").exists() and (OUT / f"{j[3]}_predictions.csv").exists())
    ]
    print("MISSING_SPATIAL", len(missing), "/", len(jobs), flush=True)
    if not missing:
        return
    procs = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__)), "spatial", "worker", str(wid)], cwd=str(OUT)
        )
        for wid in (0, 1)
    ]
    rc = [p.wait() for p in procs]
    if any(rc):
        raise SystemExit(f"spatial worker failed: {rc}")


def summarize_spatial():
    TAB.mkdir(exist_ok=True)
    preds = []
    raw_rows = []
    for fold in ["spatial_A_to_B", "spatial_B_to_A"]:
        for method in list(SPATIAL_METHODS) + ["baseline_ce_seedavg", "ordprog_full_seedavg", "BEST_blend"]:
            # published Table3 rows handled below
            pass
        for method in SPATIAL_METHODS:
            for seed in SEEDS:
                tag = f"{fold}__{method}__seed{seed}"
                path = OUT / f"{tag}.json"
                if not path.exists():
                    continue
                raw_rows.append(json.loads(path.read_text()))
                preds.append(pd.read_csv(OUT / f"{tag}_predictions.csv"))

    if raw_rows:
        pd.DataFrame(raw_rows).to_csv(TAB / "audit_ord_spatial_metrics_raw.csv", index=False)

    out_rows = []
    if preds:
        allp = pd.concat(preds, ignore_index=True)

        def seedavg_model(fold, model):
            sub = allp[(allp.fold == fold) & (allp.model == model)]
            return (
                sub.pivot_table(
                    index=["stage", "pass_id", "pass_index"], columns="seed", values="prediction"
                )
                .mean(axis=1)
                .rename("prediction")
                .reset_index()
            )

        for fold in ["spatial_A_to_B", "spatial_B_to_A"]:
            for method in SPATIAL_METHODS:
                tmp = seedavg_model(fold, method)
                if tmp.empty:
                    continue
                y = tmp.stage.to_numpy()
                p = tmp.prediction.to_numpy()
                y3 = np.where(y <= 40, 0, np.where(y <= 110, 1, 2))
                p3 = np.where(p <= 40, 0, np.where(p <= 110, 1, 2)).astype(int)
                out_rows.append(
                    dict(
                        fold=fold,
                        method=f"{method}_seedavg",
                        mae_pass=float(np.abs(y - p).mean()),
                        mae_stage_macro=stage_macro_mae(y, p),
                        spearman=spearman(y, p),
                        acc3_from_prog=classification_acc(y3, p3),
                    )
                )

    # Append published Table3 comparators if present
    t3 = OUT / "tables/Table3_spatial.csv"
    if t3.exists():
        old = pd.read_csv(t3)
        for _, r in old.iterrows():
            out_rows.append(
                dict(
                    fold=r["fold"],
                    method=r["method"],
                    mae_pass=float(r["mae_pass"]),
                    mae_stage_macro=float(r["mae_stage_macro"]),
                    spearman=float(r["spearman"]),
                    acc3_from_prog=float(r["acc3_from_prog"]),
                )
            )

    tab = pd.DataFrame(out_rows)
    if not tab.empty:
        tab = tab.sort_values(["fold", "mae_stage_macro"])
    tab.to_csv(TAB / "audit_ord_spatial.csv", index=False)
    tab.to_csv(RELEASE / "audit_ord_spatial.csv", index=False)
    print(tab.to_string(index=False) if not tab.empty else "NO_SPATIAL", flush=True)
    return tab


def summarize_all():
    corrupt_path = TAB / "audit_ord_corrupt_summary.csv"
    spatial_path = TAB / "audit_ord_spatial.csv"
    if not corrupt_path.exists() or not spatial_path.exists():
        print("Need corrupt + spatial first", flush=True)
        return
    summ = pd.read_csv(corrupt_path, index_col=0)
    spat = pd.read_csv(spatial_path)

    # headline decision focused on aord_erm
    aord_c = summ.loc["aord_erm"] if "aord_erm" in summ.index else None
    metal_c = summ.loc["metal_op_learn"] if "metal_op_learn" in summ.index else None
    free_c = summ.loc["free_op"] if "free_op" in summ.index else None
    fwide_c = summ.loc["audit_f_wide"] if "audit_f_wide" in summ.index else None
    concat_c = summ.loc["concat_erm"] if "concat_erm" in summ.index else None

    def spat_mae(method_suffix):
        sub = spat[spat.method == method_suffix]
        if sub.empty:
            return None
        return float(sub["mae_stage_macro"].mean())

    headline = dict(
        focus="aord_erm",
        clean_mae=float(aord_c["clean"]) if aord_c is not None else None,
        mean_corrupt=float(aord_c["mean_corrupt"]) if aord_c is not None else None,
        metal_learn_mean_corrupt=float(metal_c["mean_corrupt"]) if metal_c is not None else None,
        free_mean_corrupt=float(free_c["mean_corrupt"]) if free_c is not None else None,
        floorrefine_mean_corrupt=float(fwide_c["mean_corrupt"]) if fwide_c is not None else None,
        concat_mean_corrupt=float(concat_c["mean_corrupt"]) if concat_c is not None else None,
        spatial_mean_mae=spat_mae("aord_erm_seedavg"),
        spatial_metal_learn_mean_mae=spat_mae("metal_op_learn_seedavg"),
        spatial_floorrefine_mean_mae=spat_mae("audit_f_wide_seedavg"),
        spatial_ordprog_full_mean_mae=spat_mae("ordprog_full_seedavg"),
        spatial_baseline_mean_mae=spat_mae("baseline_ce_seedavg"),
        beats_metal_on_corrupt=(
            aord_c is not None
            and metal_c is not None
            and float(aord_c["mean_corrupt"]) < float(metal_c["mean_corrupt"])
        ),
        beats_floorrefine_on_corrupt=(
            aord_c is not None
            and fwide_c is not None
            and float(aord_c["mean_corrupt"]) < float(fwide_c["mean_corrupt"])
        ),
        beats_metal_on_spatial=(
            spat_mae("aord_erm_seedavg") is not None
            and spat_mae("metal_op_learn_seedavg") is not None
            and spat_mae("aord_erm_seedavg") < spat_mae("metal_op_learn_seedavg")
        ),
        note="Corruption on LG-10 ckpts; spatial retrained A↔B. No manuscript edit.",
    )
    # fill relative spatial flags vs OrdProg if available
    if headline["spatial_mean_mae"] is not None and headline["spatial_ordprog_full_mean_mae"] is not None:
        headline["beats_ordprog_on_spatial"] = (
            headline["spatial_mean_mae"] < headline["spatial_ordprog_full_mean_mae"]
        )

    (TAB / "audit_ord_extras_headline.json").write_text(json.dumps(headline, indent=2))
    (RELEASE / "audit_ord_extras_headline.json").write_text(json.dumps(headline, indent=2))
    print(json.dumps(headline, indent=2), flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "corrupt":
        run_corruption()
        return
    if cmd == "spatial" and len(sys.argv) >= 4 and sys.argv[2] == "worker":
        _spatial_train_worker(int(sys.argv[3]))
        return
    if cmd == "spatial":
        spatial_main()
        summarize_spatial()
        return
    if cmd == "summarize":
        summarize_spatial()
        summarize_all()
        return
    if cmd == "all":
        run_corruption()
        spatial_main()
        summarize_spatial()
        summarize_all()
        return
    raise SystemExit(cmd)


if __name__ == "__main__":
    main()
