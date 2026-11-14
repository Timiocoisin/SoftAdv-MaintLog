#!/usr/bin/env python3
"""Leak-Guard-k + OPLI transfer on CrossTask (non-IMPACT public ordered source).

Primary task 40567 = Change a Tire (G=11 WikiHow steps). Videos from the
third-party WorldPrediction subset of CrossTask primary clips
(SteveZeyuZhang/CrossTask_COIN_3rd-party on HF); temporal step boundaries
from the official CrossTask release (Zhukov et al., CVPR 2019).

Protocol mirrors IMPACT transfer: ResNet-18 progress head (smooth-L1 on
stage index), Leak-Guard-k mid-gap exclusion, random 80/20 baseline.
No OrdProg / no metal / no boiler images.

Usage:
  PY=/root/anaconda3/envs/apds/bin/python
  $PY run_crosstask_leak_guard.py all
  $PY run_crosstask_leak_guard.py prepare|train|summarize [video_id]
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets/crosstask"
ANN = DATA / "release/crosstask_release/annotations"
VID_DIR = DATA / "videos/40567"
RELEASE = (
    ROOT
    / "elsarticle/release_tables"
)
FIG_DIR = (
    ROOT
    / "elsarticle/figures"
)

TASK_ID = 40567
TASK_NAME = "Change a Tire"
# Canonical WikiHow step order (CrossTask tasks_primary.txt).
STEP_NAMES = [
    "brake on",
    "get things out",
    "start loose",
    "jack up",
    "unscrew wheel",
    "withdraw wheel",
    "put wheel",
    "screw wheel",
    "jack down",
    "tight wheel",
    "put things back",
]
# Prefer shorter complete 11-step clips that we already downloaded.
VIDEOS = ["LReec2a4XeQ", "i2JWkDyg26A"]
SEEDS = [0, 1, 2]
GUARDS = [0, 1, 2, 3, 4]
MAX_EPOCHS = 12
FRAMES_PER_STAGE = 8
IMAGENET_CKPT = Path("/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")


def out_dir(vid: str) -> Path:
    return OUT_ROOT / f"crosstask_leak_{TASK_ID}_{vid}"


def stage_macro_mae(y, p):
    return float(pd.DataFrame({"g": y, "e": np.abs(y - p)}).groupby("g").e.mean().mean())


def load_intervals(vid: str) -> list[tuple[int, float, float]]:
    path = ANN / f"{TASK_ID}_{vid}.csv"
    rows = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split(",")
        if len(parts) < 3:
            continue
        s, a0, a1 = int(float(parts[0])), float(parts[1]), float(parts[2])
        if s == 99:
            continue
        rows.append((s, a0, a1))
    return rows


def prepare_one(vid: str):
    import cv2

    video = VID_DIR / f"{vid}.mp4"
    if not video.exists():
        raise FileNotFoundError(video)
    intervals = load_intervals(vid)
    steps = sorted({s for s, _, _ in intervals})
    if not steps:
        raise RuntimeError(f"no steps for {vid}")
    # Remap present steps to contiguous ordered stages 1..G preserving order.
    step2g = {s: i + 1 for i, s in enumerate(steps)}
    order = [STEP_NAMES[s - 1] if 1 <= s <= len(STEP_NAMES) else f"step{s}" for s in steps]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Collect candidate frame indices per stage from annotated intervals.
    by_stage: dict[int, list[int]] = {g: [] for g in step2g.values()}
    for s, a0, a1 in intervals:
        g = step2g[s]
        i0 = max(0, int(round(a0 * fps)))
        i1 = min(n_vid - 1, int(round(a1 * fps)))
        if i1 < i0:
            i0, i1 = i1, i0
        # denser than endpoints so short intervals still yield samples
        n_take = max(FRAMES_PER_STAGE * 2, i1 - i0 + 1)
        idxs = np.linspace(i0, i1, num=min(n_take, max(1, i1 - i0 + 1))).round().astype(int)
        by_stage[g].extend(idxs.tolist())
    for g in by_stage:
        by_stage[g] = sorted(set(by_stage[g]))

    rng = np.random.default_rng(0)
    keep_idx, keep_g = [], []
    for g in sorted(by_stage):
        idxs = np.asarray(by_stage[g], dtype=np.int64)
        if len(idxs) == 0:
            continue
        take = min(FRAMES_PER_STAGE, len(idxs))
        chosen = np.linspace(0, len(idxs) - 1, take).round().astype(int)
        chosen = np.unique(idxs[chosen])
        if len(chosen) < take:
            extra = rng.choice(idxs, size=take - len(chosen), replace=True)
            chosen = np.unique(np.concatenate([chosen, extra]))
        keep_idx.extend(chosen.tolist())
        keep_g.extend([g] * len(chosen))
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    keep_g = np.asarray(keep_g, dtype=np.int32)

    images = []
    ok_g, ok_i = [], []
    for i, g in zip(keep_idx, keep_g):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        images.append(frame)
        ok_g.append(int(g))
        ok_i.append(int(i))
    cap.release()
    images = np.stack(images, 0).astype(np.uint8)
    keep_g = np.asarray(ok_g, dtype=np.int32)
    keep_idx = np.asarray(ok_i, dtype=np.int64)

    out = out_dir(vid)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", images)
    np.savez_compressed(
        out / "meta.npz",
        stages=keep_g,
        frame_idx=keep_idx,
        order=np.asarray(order),
        raw_steps=np.asarray(steps),
    )
    meta = dict(
        dataset="CrossTask",
        task_id=TASK_ID,
        task_name=TASK_NAME,
        video_id=vid,
        video=str(video),
        G=int(len(order)),
        n_sampled=int(len(images)),
        frames_per_stage=FRAMES_PER_STAGE,
        stage_order=order,
        fps=fps,
        n_vid_frames=n_vid,
        citation=(
            "CrossTask (Zhukov et al., CVPR 2019); video via third-party "
            "WorldPrediction subset (SteveZeyuZhang/CrossTask_COIN_3rd-party); "
            "annotations from official crosstask_release"
        ),
        protocol=(
            "Leak-Guard-k transfer on non-IMPACT ordered source; "
            "WikiHow step IDs as ordered stages; ResNet-18 progress head only"
        ),
    )
    (out / "protocol.json").write_text(json.dumps(meta, indent=2))
    print("WROTE", out, images.shape, "G", meta["G"], "steps", steps, flush=True)


def build_folds(stages: np.ndarray, guards):
    G = int(stages.max())
    centres = [max(2, min(G - 1, G // 2))]
    half = 1
    folds = []
    for k in guards:
        core = np.zeros(len(stages), dtype=bool)
        excluded = np.zeros(len(stages), dtype=bool)
        for c in centres:
            core |= (stages >= c - half) & (stages <= c + half)
            excluded |= (stages >= c - half - k) & (stages <= c + half + k)
        train_stages = sorted(set(stages.tolist()) - set(stages[excluded].tolist()))
        if len(train_stages) < 2:
            continue
        folds.append(
            dict(
                name=f"ct_guard{k}",
                guard=k,
                centres=centres,
                half=half,
                train_stages=train_stages,
                core_stages=sorted(set(stages[core].tolist())),
                n_train=len(train_stages),
                n_core=len(set(stages[core].tolist())),
            )
        )
    return folds


def train_one(vid: str):
    import torch
    import torch.nn as nn
    from torchvision import models

    out = out_dir(vid)
    torch.set_num_threads(2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = (
        torch.from_numpy(np.load(out / "images.npy")).permute(0, 3, 1, 2).contiguous().to(device)
    )
    meta = np.load(out / "meta.npz", allow_pickle=True)
    stages = meta["stages"].astype(np.int64)
    folds = build_folds(stages, GUARDS)
    (out / "folds.json").write_text(json.dumps(folds, indent=2))

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    weight = torch.load(IMAGENET_CKPT, map_location="cpu", weights_only=True)

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    def make_model(seed):
        seed_all(seed)
        m = models.resnet18(weights=None)
        m.load_state_dict(weight)
        m.fc = nn.Linear(512, 1)
        return m.to(device)

    def normalize(x, augment=False):
        x = x.float() / 255.0
        if augment:
            flip = torch.rand(len(x), device=device) < 0.5
            x[flip] = x[flip].flip(-1)
        return (x - mean) / std

    @torch.inference_mode()
    def predict(model, idx):
        model.eval()
        outs = []
        for batch in torch.tensor(idx, device=device).split(64):
            x = normalize(images[batch])
            outs.append(model(x).squeeze(1).cpu().numpy())
        return np.concatenate(outs)

    def train_eval(train_mask, test_mask, core_mask, seed, tag):
        result_path = out / f"{tag}_seed{seed}.json"
        pred_path = out / f"{tag}_seed{seed}_predictions.csv"
        if result_path.exists() and pred_path.exists():
            row = json.loads(result_path.read_text())
            print(
                "SKIP",
                vid,
                tag,
                seed,
                "core",
                round(float(row.get("mae_core", float("nan"))), 3),
                flush=True,
            )
            return row
        model = make_model(seed)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        y = torch.tensor(stages.astype(np.float32), device=device)
        train_idx = torch.tensor(np.flatnonzero(train_mask), device=device)
        n = len(train_idx)
        perm = train_idx[torch.randperm(n)]
        n_val = max(1, n // 5)
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        if len(fit_idx) == 0:
            fit_idx = train_idx
            val_idx = train_idx[:1]
        best = (float("inf"), 1)
        best_state = None
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            order = fit_idx[torch.randperm(len(fit_idx))]
            for batch in order.split(32):
                pred = model(normalize(images[batch], True)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y[batch])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sch.step()
            pv = predict(model, val_idx.cpu().numpy())
            vm = float(np.mean(np.abs(pv - stages[val_idx.cpu().numpy()])))
            if vm < best[0]:
                best = (vm, epoch)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        assert best_state is not None
        model = make_model(seed)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=best[1])
        for epoch in range(1, best[1] + 1):
            model.train()
            order = train_idx[torch.randperm(len(train_idx))]
            for batch in order.split(32):
                pred = model(normalize(images[batch], True)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y[batch])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sch.step()
        test_idx = np.flatnonzero(test_mask)
        ph = predict(model, test_idx)
        core_m = core_mask[test_idx]
        mae_core = (
            stage_macro_mae(stages[test_idx][core_m], ph[core_m]) if core_m.any() else float("nan")
        )
        mae_all = stage_macro_mae(stages[test_idx], ph)
        row = dict(tag=tag, seed=seed, selected_epochs=best[1], mae_core=mae_core, mae_test=mae_all)
        pd.DataFrame(
            dict(stage=stages[test_idx], prediction=ph, in_core=core_m, frame_i=test_idx)
        ).to_csv(pred_path, index=False)
        result_path.write_text(json.dumps(row, indent=2))
        print("COMPLETE", vid, tag, seed, "core", round(mae_core, 3), flush=True)
        return row

    rows = []
    for fold in folds:
        train_mask = np.isin(stages, fold["train_stages"])
        core_mask = np.isin(stages, fold["core_stages"])
        test_mask = ~train_mask
        for seed in SEEDS:
            rows.append(train_eval(train_mask, test_mask, core_mask, seed, fold["name"]))
    for seed in SEEDS:
        seed_all(seed)
        n = len(stages)
        perm = np.random.permutation(n)
        n_te = max(1, n // 5)
        te, tr = perm[:n_te], perm[n_te:]
        train_mask = np.zeros(n, dtype=bool)
        train_mask[tr] = True
        test_mask = np.zeros(n, dtype=bool)
        test_mask[te] = True
        rows.append(train_eval(train_mask, test_mask, test_mask.copy(), seed, "ct_random80"))

    pd.DataFrame(rows).to_csv(out / "raw_results.csv", index=False)
    print("WROTE raw_results", vid, flush=True)


def summarize_one(vid: str) -> pd.DataFrame:
    out = out_dir(vid)
    raw = pd.read_csv(out / "raw_results.csv")
    folds = json.loads((out / "folds.json").read_text())
    meta = json.loads((out / "protocol.json").read_text())
    board = []
    for fold in folds:
        name = fold["name"]
        preds = []
        stages = None
        for seed in SEEDS:
            p = pd.read_csv(out / f"{name}_seed{seed}_predictions.csv")
            p = p[p["in_core"]].sort_values("frame_i")
            preds.append(p["prediction"].values)
            stages = p["stage"].values
        pred = np.mean(np.stack(preds, 0), 0)
        board.append(
            dict(
                split=name,
                guard=fold["guard"],
                n_train_stages=fold["n_train"],
                n_core_stages=fold["n_core"],
                mae_seedavg=stage_macro_mae(stages, pred),
                dataset="CrossTask",
                trial=f"{TASK_ID}_{vid}",
                video_id=vid,
                task=TASK_NAME,
            )
        )
    rand = raw[raw["tag"] == "ct_random80"]
    board.append(
        dict(
            split="ct_random80",
            guard=-1,
            n_train_stages=-1,
            n_core_stages=-1,
            mae_seedavg=float(rand["mae_core"].mean()),
            dataset="CrossTask",
            trial=f"{TASK_ID}_{vid}",
            video_id=vid,
            task=TASK_NAME,
            mae_per_seed=";".join(f"{x:.3f}" for x in rand["mae_core"].tolist()),
        )
    )
    board_df = pd.DataFrame(board)
    board_df.to_csv(out / "summary.csv", index=False)
    board_df.to_csv(out / "crosstask_leak_guard_summary.csv", index=False)

    rand_mae = float(board_df.loc[board_df["guard"] < 0, "mae_seedavg"].iloc[0])
    primary = board_df[board_df["guard"] >= 0].sort_values("guard")
    viable = primary[primary["n_train_stages"] >= 3]
    if len(viable) == 0:
        viable = primary[primary["n_train_stages"] >= 2]
    kstar_row = viable.loc[viable["mae_seedavg"].idxmax()]
    infl = float(kstar_row["mae_seedavg"] / max(rand_mae, 1e-6))
    summary = dict(
        dataset="CrossTask",
        task_id=TASK_ID,
        task_name=TASK_NAME,
        video_id=vid,
        G=meta["G"],
        random_mae=rand_mae,
        primary_guard=int(kstar_row["guard"]),
        primary_mae=float(kstar_row["mae_seedavg"]),
        inflation=infl,
        note="ResNet-18 progress head; non-IMPACT public ordered procedure (Change a Tire)",
    )
    (out / "headline.json").write_text(json.dumps(summary, indent=2))
    print(board_df.to_string(index=False))
    print(summary)
    return board_df


def summarize_all(vids: list[str]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    boards = []
    headlines = []
    for vid in vids:
        boards.append(summarize_one(vid))
        headlines.append(json.loads((out_dir(vid) / "headline.json").read_text()))
    all_board = pd.concat(boards, ignore_index=True)
    RELEASE.mkdir(parents=True, exist_ok=True)
    all_board.to_csv(RELEASE / "crosstask_leak_guard_summary.csv", index=False)
    (RELEASE / "crosstask_leak_guard_headline.json").write_text(
        json.dumps(headlines, indent=2)
    )

    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    colors = ["#009E73", "#CC79A7"]
    for i, vid in enumerate(vids):
        b = boards[i]
        gdf = b[b["guard"] >= 0].sort_values("guard")
        rand_mae = float(b.loc[b["guard"] < 0, "mae_seedavg"].iloc[0])
        ax.plot(
            gdf["guard"],
            gdf["mae_seedavg"],
            "o-",
            color=colors[i % len(colors)],
            lw=1.5,
            label=f"{vid[:6]}…",
        )
        ax.scatter([-1 - 0.05 * i], [rand_mae], marker="*", s=60, color=colors[i % len(colors)])
    ax.set_xticks([-1] + sorted(set(all_board.loc[all_board["guard"] >= 0, "guard"].astype(int))))
    ax.set_xticklabels(
        ["rand"]
        + [str(x) for x in sorted(set(all_board.loc[all_board["guard"] >= 0, "guard"].astype(int)))]
    )
    ax.set_xlabel("Guard margin k (CrossTask Change-a-Tire)")
    ax.set_ylabel("Stage-macro MAE")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "Fig14_crosstask_leak_guard.pdf")
    fig.savefig(FIG_DIR / "Fig14_crosstask_leak_guard.png", dpi=600)
    plt.close(fig)
    print("FIG", FIG_DIR / "Fig14_crosstask_leak_guard.pdf")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("prepare|train|summarize|all [video_id]")
    cmd = sys.argv[1]
    vids = [sys.argv[2]] if len(sys.argv) > 2 else list(VIDEOS)
    if cmd == "prepare":
        for v in vids:
            prepare_one(v)
    elif cmd == "train":
        for v in vids:
            train_one(v)
    elif cmd == "summarize":
        summarize_all(vids)
    elif cmd == "all":
        for v in vids:
            prepare_one(v)
            train_one(v)
        summarize_all(vids)
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main()
