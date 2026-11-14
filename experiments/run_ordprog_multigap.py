#!/usr/bin/env python3
"""P0 multi-gap runner for OrdProg-Net (SoftAdv protocol).

Usage:
  python run_ordprog_multigap.py prepare
  python run_ordprog_multigap.py main          # baselines + 2 GPU workers
  python run_ordprog_multigap.py worker 0
  python run_ordprog_multigap.py summarize
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

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from training.ordprog_net import (  # noqa: E402
    VARIANTS,
    OrdProgNet,
    decode_predictions,
    ordprog_loss,
)

MAX_EPOCHS = 16
SEEDS = [0, 1, 2]
VARIANT_NAMES = list(VARIANTS.keys())
IMAGENET_CKPT = Path("/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")


def stage_macro_mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(pd.DataFrame({"g": y, "e": np.abs(y - p)}).groupby("g").e.mean().mean())


def classification_acc(y: np.ndarray, p: np.ndarray) -> float:
    return float((y == p).mean()) if len(y) else float("nan")


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    return float(pd.Series(y).corr(pd.Series(p), method="spearman"))


def load_pass_arrays():
    a = np.load(OUT / "sampled_data.npz")
    return a["handcrafted"], a["embedding"], a["stages"], a["keys"]


def build_multi_gap_folds(g: np.ndarray) -> list:
    centers = [25, 55, 85, 115, 145]
    folds = []
    for guard in [0, 10]:
        excluded = np.zeros(len(g), dtype=bool)
        core = np.zeros(len(g), dtype=bool)
        for c in centers:
            core |= (g >= c - 4) & (g <= c + 4)
            excluded |= (g >= c - 4 - guard) & (g <= c + 4 + guard)
        train_stages = np.unique(g[~excluded])
        # rank-based inner val: every 5th stage, offset 2
        val = train_stages[np.arange(len(train_stages)) % 5 == 2]
        fit = np.setdiff1d(train_stages, val)
        core_stages = np.unique(g[core])
        fold = {
            "name": f"multi_gap_guard{guard}",
            "kind": "multi_gap",
            "center": -1,
            "guard": guard,
            "train_stages": train_stages.tolist(),
            "test_stages": np.unique(g[excluded]).tolist(),
            "core_test_stages": core_stages.tolist(),
            "inner_fit_stages": fit.tolist(),
            "inner_val_stages": val.tolist(),
        }
        assert len(core_stages) == 45
        assert not set(train_stages) & set(fold["test_stages"])
        assert len(val) >= 3, fold["name"]
        folds.append(fold)
    return folds


def prepare():
    assert (OUT / "sampled_images.npy").exists(), "missing sampled_images.npy link"
    assert (OUT / "sampled_data.npz").exists(), "missing sampled_data.npz link"
    h, e, g, keys = load_pass_arrays()
    assert len(g) == 877
    folds = build_multi_gap_folds(g)
    protocol = json.loads((OUT / "protocol.json").read_text())
    protocol["folds"] = folds
    protocol["seeds"] = SEEDS
    protocol["max_epochs"] = MAX_EPOCHS
    protocol["variants"] = {k: v.to_dict() for k, v in VARIANTS.items()}
    protocol["prepared_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    # empty / template tables
    pd.DataFrame(
        columns=[
            "fold",
            "model",
            "seed",
            "guard",
            "selected_epochs",
            "inner_val_mae",
            "mae_pass",
            "mae_stage_macro",
            "mae_pass_core45",
            "mae_stage_macro_core45",
            "spearman_pass",
            "acc3_pass",
            "acc5_pass",
            "acc3_pass_core45",
            "acc5_pass_core45",
            "elapsed_seconds",
            "checkpoint_sha256",
        ]
    ).to_csv(OUT / "tables/Table2_multi_gap_template.csv", index=False)
    pd.DataFrame(
        {
            "variant": VARIANT_NAMES,
            "use_appearance": [VARIANTS[v].use_appearance for v in VARIANT_NAMES],
            "use_coral": [VARIANTS[v].use_coral for v in VARIANT_NAMES],
            "role": [
                "baseline skeleton (CE3+CE5+progress)",
                "module A only (CORAL+progress)",
                "module B only (appearance+CE)",
                "full OrdProg-Net (A+B)",
            ],
        }
    ).to_csv(OUT / "tables/Table6_ablation_plan.csv", index=False)
    print("PREPARED", OUT / "protocol.json", "folds", [f["name"] for f in folds], flush=True)
    return protocol


def baseline_run(protocol):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    h, e, g, keys = load_pass_arrays()
    rows, predictions = [], []
    candidates = [
        ("handcrafted_rf", h, [8, 16, None]),
        ("handcrafted_knn", h, [3, 5, 9]),
        ("imagenet_knn", e, [3, 5, 9]),
        ("imagenet_ridge", e, [1.0, 10.0, 100.0]),
    ]
    for fold in protocol["folds"]:
        fit = np.isin(g, fold["inner_fit_stages"])
        val = np.isin(g, fold["inner_val_stages"])
        train = fit | val
        test = np.isin(g, fold["test_stages"])
        core = np.isin(g, fold["core_test_stages"])
        for name, x, params in candidates:
            best = None
            for param in params:
                scaler = StandardScaler().fit(x[fit])
                a, b = scaler.transform(x[fit]), scaler.transform(x[val])
                if name.endswith("rf"):
                    est = RandomForestRegressor(
                        n_estimators=400, max_depth=param, random_state=42, n_jobs=2
                    )
                elif name.endswith("knn"):
                    est = KNeighborsRegressor(n_neighbors=param, weights="distance")
                else:
                    est = Ridge(alpha=param)
                est.fit(a, g[fit] / 169.0)
                ph = est.predict(b) * 169.0
                metric = stage_macro_mae(g[val], ph)
                if best is None or metric < best[0]:
                    best = (metric, param)
            scaler = StandardScaler().fit(x[train])
            a, b = scaler.transform(x[train]), scaler.transform(x[test])
            if name.endswith("rf"):
                est = RandomForestRegressor(
                    n_estimators=400, max_depth=best[1], random_state=42, n_jobs=2
                )
            elif name.endswith("knn"):
                est = KNeighborsRegressor(n_neighbors=best[1], weights="distance")
            else:
                est = Ridge(alpha=best[1])
            est.fit(a, g[train] / 169.0)
            ph = est.predict(b) * 169.0
            test_idx = np.flatnonzero(test)
            core_mask = core[test]
            rows.append(
                dict(
                    fold=fold["name"],
                    model=name,
                    seed=42,
                    guard=fold["guard"],
                    selected_parameter=best[1],
                    inner_val_mae=best[0],
                    mae_pass=float(np.abs(ph - g[test]).mean()),
                    mae_stage_macro=stage_macro_mae(g[test], ph),
                    mae_pass_core45=float(np.abs(ph[core_mask] - g[test][core_mask]).mean()),
                    mae_stage_macro_core45=stage_macro_mae(g[test][core_mask], ph[core_mask]),
                    spearman_pass=spearman(g[test], ph),
                )
            )
            for j, idx in enumerate(test_idx):
                predictions.append(
                    dict(
                        fold=fold["name"],
                        model=name,
                        seed=42,
                        stage=int(g[idx]),
                        pass_id=int(keys[idx, 1]),
                        prediction=float(ph[j]),
                        error=float(abs(ph[j] - g[idx])),
                        in_core45=bool(core[idx]),
                    )
                )
        print("BASELINES", fold["name"], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "baseline_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "baseline_predictions.csv", index=False)


def cnn_worker(worker_id: int):
    import torch
    from torch import nn

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.set_device(worker_id)
    device = torch.device(f"cuda:{worker_id}")

    protocol = json.loads((OUT / "protocol.json").read_text())
    h, e, g, keys = load_pass_arrays()
    images = (
        torch.from_numpy(np.load(OUT / "sampled_images.npy").copy())
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device)
    )
    gp = np.repeat(g, 3)
    targets = torch.tensor(gp / 169.0, dtype=torch.float32, device=device)
    c3 = torch.tensor(
        np.where(gp <= 40, 0, np.where(gp <= 110, 1, 2)), device=device, dtype=torch.long
    )
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

    def make_net(variant: str, seed: int):
        seed_all(seed)
        cfg = VARIANTS[variant]
        model = OrdProgNet(cfg, imagenet_state=weight).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        return model, opt, sch, cfg

    def train_epoch(model, opt, cfg, ids):
        model.train()
        order = ids[torch.randperm(len(ids), device=device)]
        total = 0.0
        for batch in order.split(64):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o3, o5, op = model(normalize(batch, True), mean, std)
                loss = ordprog_loss(o3, o5, op, c3[batch], c5[batch], targets[batch], cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
        return total / len(ids)

    @torch.inference_mode()
    def predict_pass(model, cfg, pass_mask):
        ids = torch.tensor(np.flatnonzero(np.repeat(pass_mask, 3)), device=device)
        model.eval()
        prog_parts, y3_parts, y5_parts = [], [], []
        for batch in ids.split(96):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o3, o5, op = model(normalize(batch), mean, std)
            dec = decode_predictions(o3.float(), o5.float(), op.float(), cfg)
            prog_parts.append(dec["progress"].cpu().numpy())
            y3_parts.append(dec["y3"].cpu().numpy())
            y5_parts.append(dec["y5"].cpu().numpy())
        prog = np.concatenate(prog_parts).reshape(-1, 3).mean(1) * 169.0
        # majority vote over 3 frames for discrete labels
        y3 = (
            pd.DataFrame(np.concatenate(y3_parts).reshape(-1, 3))
            .mode(axis=1)[0]
            .to_numpy()
            .astype(int)
        )
        y5 = (
            pd.DataFrame(np.concatenate(y5_parts).reshape(-1, 3))
            .mode(axis=1)[0]
            .to_numpy()
            .astype(int)
        )
        return prog, y3, y5

    jobs = [
        (fold, variant, seed)
        for fold in protocol["folds"]
        for variant in VARIANT_NAMES
        for seed in SEEDS
    ]
    for job_id, (fold, variant, seed) in enumerate(jobs):
        if job_id % 2 != worker_id:
            continue
        tag = f"{fold['name']}__{variant}__seed{seed}"
        result_path = OUT / f"{tag}.json"
        if result_path.exists():
            print("SKIP_COMPLETE", tag, flush=True)
            continue
        fit = np.isin(g, fold["inner_fit_stages"])
        val = np.isin(g, fold["inner_val_stages"])
        train = fit | val
        test = np.isin(g, fold["test_stages"])
        core = np.isin(g, fold["core_test_stages"])
        assert not set(g[train]) & set(g[test])
        fitids = torch.tensor(np.flatnonzero(np.repeat(fit, 3)), device=device)
        trainids = torch.tensor(np.flatnonzero(np.repeat(train, 3)), device=device)

        start = time.time()
        model, opt, sch, cfg = make_net(variant, seed)
        history = []
        best = (float("inf"), None)
        for epoch in range(1, MAX_EPOCHS + 1):
            loss = train_epoch(model, opt, cfg, fitids)
            sch.step()
            ph, _, _ = predict_pass(model, cfg, val)
            vm = stage_macro_mae(g[val], ph)
            history.append(dict(epoch=epoch, loss=loss, val_mae=vm))
            if vm < best[0]:
                best = (vm, epoch)
            if epoch % 4 == 0:
                print(
                    "SELECT",
                    worker_id,
                    tag,
                    epoch,
                    round(vm, 3),
                    "best",
                    best[1],
                    flush=True,
                )
        del model, opt, sch
        torch.cuda.empty_cache()

        model, opt, sch, cfg = make_net(variant, seed)
        for epoch in range(1, best[1] + 1):
            train_epoch(model, opt, cfg, trainids)
            sch.step()
        ph, p3, p5 = predict_pass(model, cfg, test)
        y3_true = np.where(g[test] <= 40, 0, np.where(g[test] <= 110, 1, 2))
        y5_true = np.minimum((g[test] - 1) // 34, 4)
        core_m = core[test]

        checkpoint = OUT / f"{tag}.pt"
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "fold": fold,
                "variant": variant,
                "cfg": cfg.to_dict(),
                "seed": seed,
                "epochs": best[1],
                "selection": "inner stage validation only",
            },
            checkpoint,
        )
        predrows = [
            dict(
                fold=fold["name"],
                model=variant,
                seed=seed,
                stage=int(g[idx]),
                pass_id=int(keys[idx, 1]),
                prediction=float(ph[j]),
                pred_y3=int(p3[j]),
                pred_y5=int(p5[j]),
                error=float(abs(ph[j] - g[idx])),
                in_core45=bool(core[idx]),
            )
            for j, idx in enumerate(np.flatnonzero(test))
        ]
        pd.DataFrame(predrows).to_csv(OUT / f"{tag}_predictions.csv", index=False)
        row = dict(
            fold=fold["name"],
            model=variant,
            seed=seed,
            guard=fold["guard"],
            selected_epochs=best[1],
            inner_val_mae=best[0],
            mae_pass=float(np.abs(ph - g[test]).mean()),
            mae_stage_macro=stage_macro_mae(g[test], ph),
            mae_pass_core45=float(np.abs(ph[core_m] - g[test][core_m]).mean()),
            mae_stage_macro_core45=stage_macro_mae(g[test][core_m], ph[core_m]),
            spearman_pass=spearman(g[test], ph),
            acc3_pass=classification_acc(y3_true, p3),
            acc5_pass=classification_acc(y5_true, p5),
            acc3_pass_core45=classification_acc(y3_true[core_m], p3[core_m]),
            acc5_pass_core45=classification_acc(y5_true[core_m], p5[core_m]),
            elapsed_seconds=time.time() - start,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            history=history,
        )
        result_path.write_text(json.dumps(row, indent=2))
        print(
            "COMPLETE_CNN",
            worker_id,
            tag,
            "epochs",
            best[1],
            "macro_MAE",
            round(row["mae_stage_macro"], 3),
            "core45",
            round(row["mae_stage_macro_core45"], 3),
            "seconds",
            round(row["elapsed_seconds"], 1),
            flush=True,
        )
        del model, opt, sch
        torch.cuda.empty_cache()


def summarize():
    protocol = json.loads((OUT / "protocol.json").read_text())
    rows, preds = [], []
    missing = []
    for fold in protocol["folds"]:
        for variant in VARIANT_NAMES:
            for seed in SEEDS:
                tag = f"{fold['name']}__{variant}__seed{seed}"
                path = OUT / f"{tag}.json"
                if not path.exists():
                    missing.append(tag)
                    continue
                row = json.loads(path.read_text())
                row.pop("history", None)
                rows.append(row)
                preds.append(pd.read_csv(OUT / f"{tag}_predictions.csv"))
    if missing:
        print("MISSING", len(missing), missing[:10], flush=True)
    if not rows:
        print("NO_ROWS", flush=True)
        return
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "cnn_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(OUT / "cnn_predictions.csv", index=False)

    # Table 2 style: mean±std over seeds, prefer core45 for fair guard comparison
    agg_cols = [
        "mae_pass_core45",
        "mae_stage_macro_core45",
        "spearman_pass",
        "acc3_pass_core45",
        "acc5_pass_core45",
    ]
    summary = (
        metrics.groupby(["fold", "model"], as_index=False)[agg_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    # flatten columns
    summary.columns = [
        "_".join([c for c in col if c]).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(OUT / "tables/Table2_multi_gap_meanstd.csv", index=False)

    # Ablation on guard0 core45
    abl = metrics[metrics["fold"] == "multi_gap_guard0"].groupby("model", as_index=False)[
        agg_cols
    ].mean()
    abl.to_csv(OUT / "tables/Table6_ablation_guard0_core45.csv", index=False)
    print(abl.to_string(index=False), flush=True)
    (OUT / "COMPLETE_MULTIGAP").write_text(
        f"CNN jobs done: {len(rows)}; missing: {len(missing)}\n"
    )


def main():
    protocol = prepare()
    procs = []
    for worker in [0, 1]:
        log = open(OUT / "logs" / f"worker{worker}.log", "a", buffering=1)
        env = dict(os.environ)
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        procs.append(
            (
                subprocess.Popen(
                    [sys.executable, "-u", __file__, "worker", str(worker)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                ),
                log,
            )
        )
    baseline_run(protocol)
    total = len(protocol["folds"]) * len(VARIANT_NAMES) * len(SEEDS)
    while any(p.poll() is None for p, _ in procs):
        done = len(list(OUT.glob("multi_gap_guard*__*__seed*.json")))
        print("PROGRESS", done, "/", total, flush=True)
        time.sleep(20)
    codes = [p.wait() for p, _ in procs]
    for _, log in procs:
        log.close()
    if any(c != 0 for c in codes):
        raise RuntimeError(f"workers failed: {codes}; see logs/")
    summarize()
    print("ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "main"
    if cmd == "prepare":
        prepare()
    elif cmd == "worker":
        cnn_worker(int(sys.argv[2]))
    elif cmd == "summarize":
        summarize()
    elif cmd == "baselines":
        protocol = prepare()
        baseline_run(protocol)
    elif cmd == "main":
        main()
    else:
        raise SystemExit(f"unknown command {cmd}")
