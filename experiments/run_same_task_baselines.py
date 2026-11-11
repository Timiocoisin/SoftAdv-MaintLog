#!/usr/bin/env python3
"""Same-task (ordinal / dense-progress / temporal) baselines under Leak-Guard-10.

Trains RGB dense progress (`prog_only`) with the matched multi-gap budget, reuses
existing CORAL-progress and Temporal-TF checkpoints, and freezes manuscript boards
that separate capacity RGB from true same-task competitors.

Usage:
  /root/anaconda3/envs/apds/bin/python -u run_same_task_baselines.py train
  /root/anaconda3/envs/apds/bin/python -u run_same_task_baselines.py summarize
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
RELEASE = (
    ROOT
    / "IET_Image_Processing_LaTeX_Template"
    / "WileyDesign"
    / "Optimal-Design-layout"
    / "release_tables"
)
sys.path.insert(0, str(ROOT))

from training.ordprog_net import (  # noqa: E402
    VARIANTS,
    OrdProgNet,
    decode_predictions,
    ordprog_loss,
)

MAX_EPOCHS = 16
SEEDS = [0, 1, 2]
TRAIN_VARIANTS = ["prog_only"]  # new RGB dense-progress; others reused
IMAGENET_CKPT = Path("/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
PYTHON = "/root/anaconda3/envs/apds/bin/python"

# Provenance for manuscript board
SAME_TASK_METHODS = [
    # key, display, family, is_ours, provenance, pred_glob_g0, pred_glob_g10
    (
        "coral_prog",
        "CORAL progress (RGB)",
        "ordinal_rgb",
        False,
        "reused: multi_gap_guard{0,10}__coral_prog (Table_multi_gap / SOTA bakeoff pool)",
        "multi_gap_guard0__coral_prog__seed*_predictions.csv",
        "multi_gap_guard10__coral_prog__seed*_predictions.csv",
    ),
    (
        "prog_only",
        "Dense progress (RGB)",
        "progress_rgb",
        False,
        "trained: multi_gap_guard{0,10}__prog_only (this script; MAX_EPOCHS=16, seeds 0/1/2)",
        "multi_gap_guard0__prog_only__seed*_predictions.csv",
        "multi_gap_guard10__prog_only__seed*_predictions.csv",
    ),
    (
        "temporal_tf_r18",
        "Temporal Transformer (R18)",
        "temporal",
        False,
        "reused: sota__multi_gap_guard{0,10}__temporal_tf_r18 (run_sota_bakeoff.py)",
        "sota__multi_gap_guard0__temporal_tf_r18__seed*_predictions.csv",
        "sota__multi_gap_guard10__temporal_tf_r18__seed*_predictions.csv",
    ),
]

# Capacity / scoring rows kept on the protocol board but tagged for separation
CAPACITY_AND_OURS = [
    ("OrdProg-Net (ours)", "OrdProg-Net blend (scoring)", "ours_scoring", True),
    ("ordprog_full", "CORAL + metal (expert)", "ours_expert", True),
    ("appearance_ce", "CE + metal (expert)", "ours_expert", True),
    ("baseline_ce", "ResNet-18 CE (RGB)", "rgb_baseline", False),
    ("efficientnet_b0", "EfficientNet-B0 CE", "capacity_rgb", False),
    ("resnet50", "ResNet-50 CE", "capacity_rgb", False),
    ("convnext_tiny", "ConvNeXt-Tiny CE", "capacity_rgb", False),
    ("swin_t", "Swin-T CE", "capacity_rgb", False),
    ("vit_b_16", "ViT-B/16 CE", "capacity_rgb", False),
    ("mobilenet_v3_large", "MobileNetV3-Large CE", "capacity_rgb", False),
]


def stage_macro_mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(pd.DataFrame({"g": y, "e": np.abs(y - p)}).groupby("g").e.mean().mean())


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    return float(pd.Series(y).corr(pd.Series(p), method="spearman"))


def progress_binned_acc3(y: np.ndarray, p: np.ndarray) -> float:
    y3 = np.where(y <= 40, 0, np.where(y <= 110, 1, 2))
    p3 = np.where(p <= 40, 0, np.where(p <= 110, 1, 2))
    return float((y3 == p3).mean()) if len(y) else float("nan")


def load_pass_arrays():
    a = np.load(OUT / "sampled_data.npz")
    return a["stages"], a["keys"]


def seedavg_from_preds(pattern: str) -> dict:
    files = sorted(OUT.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    core = df[df.in_core45].copy()
    piv = core.pivot_table(index=["stage", "pass_id"], columns="seed", values="prediction")
    pred = piv.mean(axis=1).rename("prediction").reset_index()
    y = pred.stage.to_numpy()
    p = pred.prediction.to_numpy()
    return dict(
        mae=stage_macro_mae(y, p),
        spearman=spearman(y, p),
        acc3=progress_binned_acc3(y, p),
        n_seeds=int(piv.shape[1]),
        n_core=int(len(pred)),
    )


def cnn_worker(worker_id: int):
    import torch

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.set_device(worker_id)
    device = torch.device(f"cuda:{worker_id}")

    protocol = json.loads((OUT / "protocol.json").read_text())
    g, keys = load_pass_arrays()
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

    folds = [f for f in protocol["folds"] if f["name"] in ("multi_gap_guard0", "multi_gap_guard10")]
    jobs = [(fold, v, seed) for fold in folds for v in TRAIN_VARIANTS for seed in SEEDS]
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
        fitids = torch.tensor(np.flatnonzero(np.repeat(fit, 3)), device=device)
        trainids = torch.tensor(np.flatnonzero(np.repeat(train, 3)), device=device)

        start = time.time()
        model, opt, sch, cfg = make_net(variant, seed)
        history, best = [], (float("inf"), None)
        for epoch in range(1, MAX_EPOCHS + 1):
            loss = train_epoch(model, opt, cfg, fitids)
            sch.step()
            ph, _, _ = predict_pass(model, cfg, val)
            vm = stage_macro_mae(g[val], ph)
            history.append(dict(epoch=epoch, loss=loss, val_mae=vm))
            if vm < best[0]:
                best = (vm, epoch)
            if epoch % 4 == 0:
                print("SELECT", worker_id, tag, epoch, round(vm, 3), "best", best[1], flush=True)
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
            acc3_pass_core45=float((y3_true[core_m] == p3[core_m]).mean()),
            acc5_pass_core45=float((y5_true[core_m] == p5[core_m]).mean()),
            elapsed_seconds=time.time() - start,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            history=history,
        )
        result_path.write_text(json.dumps(row, indent=2))
        print(
            "COMPLETE",
            worker_id,
            tag,
            "core45",
            round(row["mae_stage_macro_core45"], 3),
            flush=True,
        )
        del model, opt, sch
        torch.cuda.empty_cache()


def train_main():
    assert "prog_only" in VARIANTS, "prog_only variant missing from training.ordprog_net"
    (OUT / "logs").mkdir(exist_ok=True)
    procs = []
    for worker in [0, 1]:
        log = open(OUT / "logs" / f"same_task_worker{worker}.log", "a", buffering=1)
        env = dict(os.environ)
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        procs.append(
            (
                subprocess.Popen(
                    [PYTHON, "-u", str(Path(__file__).resolve()), "worker", str(worker)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                ),
                log,
            )
        )
    total = 2 * len(TRAIN_VARIANTS) * len(SEEDS)
    while any(p.poll() is None for p, _ in procs):
        done = len(list(OUT.glob("multi_gap_guard*__prog_only__seed*.json")))
        print("PROGRESS_SAME_TASK", done, "/", total, flush=True)
        time.sleep(10)
    codes = [p.wait() for p, _ in procs]
    for _, log in procs:
        log.close()
    if any(c != 0 for c in codes):
        raise RuntimeError(codes)
    summarize()
    (OUT / "COMPLETE_SAME_TASK_BASELINES").write_text("ok\n")
    print("ALL_COMPLETE", flush=True)


def _paper_table2_lookup():
    src = OUT / "tables" / "Paper_Table2_main_comparison.csv"
    with src.open() as f:
        return {r["Method"]: r for r in __import__("csv").DictReader(f)}


def summarize():
    RELEASE.mkdir(parents=True, exist_ok=True)
    paper = _paper_table2_lookup()

    same_rows = []
    for key, label, family, is_ours, note, g0pat, g10pat in SAME_TASK_METHODS:
        g0 = seedavg_from_preds(g0pat)
        g10 = seedavg_from_preds(g10pat)
        same_rows.append(
            dict(
                method=label,
                source_key=key,
                family=family,
                is_ours=is_ours,
                protocol="leak_guard_kstar_10",
                mae_g0=g0["mae"],
                mae_g10=g10["mae"],
                spearman_g10=g10["spearman"],
                acc3_g10=g10["acc3"],
                ranking_primary="mae_g10",
                provenance=note,
                trained_vs_reused="trained" if key == "prog_only" else "reused",
                note="Same-task visual progress/ordinal/temporal under Leak-Guard-10; rank by MAE10 only.",
            )
        )

    same_rows.sort(key=lambda d: d["mae_g10"])
    for i, r in enumerate(same_rows, 1):
        r["rank_mae10"] = i

    same_csv = RELEASE / "same_task_baseline_board.csv"
    fields = list(same_rows[0].keys())
    with same_csv.open("w", newline="") as f:
        w = __import__("csv").DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(same_rows)

    headline = {
        "protocol": "Leak-Guard-k*=10 (same centres/core45; seed-averaged progress preds)",
        "forbidden_for_ranking": ["random_pass", "even_odd", "capacity_rgb_as_same_task"],
        "ranking_primary": "mae_g10",
        "n_methods": len(same_rows),
        "methods": [
            {
                "method": r["method"],
                "source_key": r["source_key"],
                "family": r["family"],
                "mae_g0": r["mae_g0"],
                "mae_g10": r["mae_g10"],
                "rank_mae10": r["rank_mae10"],
                "trained_vs_reused": r["trained_vs_reused"],
                "provenance": r["provenance"],
            }
            for r in same_rows
        ],
        "best_mae10": same_rows[0]["mae_g10"],
        "best_method": same_rows[0]["method"],
        "claim": (
            "Same-task baselines are ordinal/progress/temporal methods scored only under "
            "Leak-Guard-10; capacity RGB CE backbones are reported separately and are not "
            "treated as same-task SOTA competitors. Random-split MAE is leak control only."
        ),
        "not_available_in_repo": [
            "CORN / ORCNN ordinal heads (no implementation under multi_gap_guard10)",
            "TCN temporal head (no Leak-Guard-10 checkpoints)",
            "ProgressNet paper reimplementation (RGB dense progress uses local prog_only head)",
        ],
    }
    (RELEASE / "same_task_baseline_headline.json").write_text(json.dumps(headline, indent=2) + "\n")
    (OUT / "tables" / "same_task_baseline_board.csv").write_text(same_csv.read_text())
    (OUT / "tables" / "same_task_baseline_headline.json").write_text(
        json.dumps(headline, indent=2) + "\n"
    )

    # Combined protocol board: capacity vs same-task separation
    combined = []
    for key, label, family, is_ours in CAPACITY_AND_OURS:
        if key not in paper:
            continue
        r = paper[key]
        if is_ours:
            group, note = "ours", "Ours scoring/expert row under Leak-Guard-10; not an external same-task baseline."
        elif family == "capacity_rgb":
            group, note = "capacity_rgb", "Capacity RGB CE backbone; not a same-task ordinal/progress/temporal baseline."
        else:
            group, note = "rgb_baseline", "RGB CE classification baseline; not a same-task ordinal/progress/temporal baseline."
        combined.append(
            dict(
                method=label,
                source_key=key,
                family=family,
                board_group=group,
                is_ours=is_ours,
                protocol="leak_guard_kstar_10",
                mae_g0=float(r["MAE_g0"]),
                mae_g10=float(r["MAE_g10"]),
                spearman_g10=float(r["Sp_g10"]),
                acc3_g10=float(r["Acc3_g10"]),
                ranking_primary="mae_g10",
                trained_vs_reused="reused",
                note=note,
            )
        )
    for r in same_rows:
        combined.append(
            dict(
                method=r["method"],
                source_key=r["source_key"],
                family=r["family"],
                board_group="same_task",
                is_ours=r["is_ours"],
                protocol=r["protocol"],
                mae_g0=r["mae_g0"],
                mae_g10=r["mae_g10"],
                spearman_g10=r["spearman_g10"],
                acc3_g10=r["acc3_g10"],
                ranking_primary="mae_g10",
                trained_vs_reused=r["trained_vs_reused"],
                note=r["note"],
            )
        )
    combined.sort(key=lambda d: d["mae_g10"])
    for i, r in enumerate(combined, 1):
        r["rank_mae10"] = i

    proto_csv = RELEASE / "same_protocol_baseline_board.csv"
    with proto_csv.open("w", newline="") as f:
        w = __import__("csv").DictWriter(f, fieldnames=list(combined[0].keys()))
        w.writeheader()
        w.writerows(combined)

    same_best = min(same_rows, key=lambda d: d["mae_g10"])
    proto_headline = {
        "protocol": "Leak-Guard-k*=10 (same centres/core45; seed-averaged progress preds)",
        "forbidden_for_ranking": ["random_pass", "even_odd", "unmatched_backbone_without_metal"],
        "n_methods": len(combined),
        "n_same_task": len(same_rows),
        "n_capacity_rgb": sum(1 for r in combined if r["board_group"] == "capacity_rgb"),
        "best_mae10": combined[0]["mae_g10"],
        "best_method": combined[0]["method"],
        "best_same_task_mae10": same_best["mae_g10"],
        "best_same_task_method": same_best["method"],
        "rgb_ce_mae10": next(r["mae_g10"] for r in combined if r["source_key"] == "baseline_ce"),
        "board_groups": ["ours", "same_task", "rgb_baseline", "capacity_rgb"],
        "claim": (
            "External capacity RGB CE backbones and same-task ordinal/progress/temporal "
            "baselines are both scored under Leak-Guard-10, but only same_task rows answer "
            "the same-task baseline critique; random-split MAE is leak control, not ranking."
        ),
    }
    (RELEASE / "same_protocol_baseline_headline.json").write_text(
        json.dumps(proto_headline, indent=2) + "\n"
    )

    print(json.dumps(headline, indent=2), flush=True)
    print("WROTE", same_csv, proto_csv, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "worker":
        cnn_worker(int(sys.argv[2]))
    elif cmd == "summarize":
        summarize()
    elif cmd == "train":
        train_main()
    else:
        raise SystemExit(f"unknown cmd {cmd}")
