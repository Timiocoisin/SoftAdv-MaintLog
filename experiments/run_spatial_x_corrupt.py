#!/usr/bin/env python3
"""TRUE crossed region×corruption cells on spatial-fold checkpoints.

Evaluates brightness + gauss_noise (and clean reference) at eval time on
spatial A→B / B→A checkpoints. Prefer Audit-OrdNet locked blend
(0.3 aord_rob + 0.7 aord_erm); optionally also ResNet-18 CE.

No retrain: loads existing spatial_*.pt and applies _corrupt_batch before forward.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
TAB = OUT / "tables"
RELEASE = (
    ROOT
    / "elsarticle/release_tables"
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))

from training.ordprog_net import VARIANTS, build_model  # noqa: E402
import run_p0_extras as extras  # noqa: E402

SEEDS = [0, 1, 2]
IMAGENET_CKPT = Path("/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
BATCH = 64
BLEND_W_ROB = 0.3  # locked: 0.3 rob + 0.7 erm

METHODS = {
    "aord_erm": dict(variant="audit_ord_coral", tag="{fold}__aord_erm__seed{seed}"),
    "aord_rob": dict(variant="audit_ord_coral", tag="{fold}__aord_rob__seed{seed}"),
    "baseline_ce": dict(variant="baseline_ce", tag="{fold}__baseline_ce__seed{seed}"),
}


def stage_macro_mae(y, p):
    return float(pd.DataFrame({"g": y, "e": np.abs(y - p)}).groupby("g").e.mean().mean())


def load_pass_arrays():
    a = np.load(OUT / "sampled_data.npz")
    return a["stages"], a["keys"]


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    proto = json.loads((OUT / "protocol_extras.json").read_text())
    folds = {f["name"]: f for f in proto["spatial_folds"]}
    fams = proto["corruption"]["families"]
    # brightness + noise full grids + clean
    eval_families = {
        "clean": [None],
        "brightness": list(fams["brightness"]),
        "gauss_noise_sigma": list(fams["gauss_noise_sigma"]),
    }

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

    def load_model(method: str, fold: str, seed: int):
        spec = METHODS[method]
        cfg = VARIANTS[spec["variant"]]
        tag = spec["tag"].format(fold=fold, seed=seed)
        path = OUT / f"{tag}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        model = build_model(cfg, imagenet_state=weight).to(device)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        return model, tag

    @torch.inference_mode()
    def predict_pass(model, pass_indices, family, severity):
        mask = np.zeros(len(g), dtype=bool)
        mask[np.asarray(pass_indices)] = True
        ids = torch.tensor(np.flatnonzero(np.repeat(mask, 3)), device=device)
        parts = []
        for batch in ids.split(BATCH):
            x = images[batch].float() / 255.0
            if family != "clean":
                # gauss_noise is stochastic — seed by batch start for reproducibility
                if family == "gauss_noise_sigma":
                    torch.manual_seed(int(batch[0].item()) + 17)
                x = extras._corrupt_batch(x, family, severity, device)
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                op = model(x, mean, std)[2]
            parts.append(op.float().cpu().numpy())
        return np.concatenate(parts).reshape(-1, 3).mean(1) * 169.0

    raw_rows = []
    # Cache per (method, fold, seed, family, sev) predictions keyed by pass order
    pred_cache: dict[tuple, np.ndarray] = {}

    for fold_name, fold in folds.items():
        test_passes = np.array(fold["test_pass_indices"])
        y = g[test_passes]
        for family, sevs in eval_families.items():
            for sev in sevs:
                print(f"EVAL {fold_name} {family} {sev}", flush=True)
                for method in METHODS:
                    seed_preds = []
                    for seed in SEEDS:
                        key = (method, fold_name, seed, family, sev)
                        if key not in pred_cache:
                            model, tag = load_model(method, fold_name, seed)
                            pred_cache[key] = predict_pass(model, test_passes, family, sev)
                            del model
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                        p = pred_cache[key]
                        seed_preds.append(p)
                        raw_rows.append(
                            dict(
                                fold=fold_name,
                                method=method,
                                seed=seed,
                                corruption=family,
                                severity="" if sev is None else float(sev),
                                mae_stage_macro=stage_macro_mae(y, p),
                                mae_pass=float(np.mean(np.abs(p - y))),
                                n_test_passes=int(len(test_passes)),
                                crossed_region_x_corrupt=True,
                                checkpoint_fold=fold_name,
                            )
                        )
                    # seedavg
                    p_avg = np.mean(np.stack(seed_preds, 0), 0)
                    raw_rows.append(
                        dict(
                            fold=fold_name,
                            method=f"{method}_seedavg",
                            seed=-1,
                            corruption=family,
                            severity="" if sev is None else float(sev),
                            mae_stage_macro=stage_macro_mae(y, p_avg),
                            mae_pass=float(np.mean(np.abs(p_avg - y))),
                            n_test_passes=int(len(test_passes)),
                            crossed_region_x_corrupt=True,
                            checkpoint_fold=fold_name,
                        )
                    )

                # Locked blend on seedavg parents
                p_rob = np.mean(
                    np.stack(
                        [pred_cache[("aord_rob", fold_name, s, family, sev)] for s in SEEDS],
                        0,
                    ),
                    0,
                )
                p_erm = np.mean(
                    np.stack(
                        [pred_cache[("aord_erm", fold_name, s, family, sev)] for s in SEEDS],
                        0,
                    ),
                    0,
                )
                p_blend = BLEND_W_ROB * p_rob + (1.0 - BLEND_W_ROB) * p_erm
                raw_rows.append(
                    dict(
                        fold=fold_name,
                        method="blend_locked_0.3rob+0.7erm",
                        seed=-1,
                        corruption=family,
                        severity="" if sev is None else float(sev),
                        mae_stage_macro=stage_macro_mae(y, p_blend),
                        mae_pass=float(np.mean(np.abs(p_blend - y))),
                        n_test_passes=int(len(test_passes)),
                        crossed_region_x_corrupt=True,
                        checkpoint_fold=fold_name,
                        note="0.3*seedavg(aord_rob)+0.7*seedavg(aord_erm)",
                    )
                )

    raw = pd.DataFrame(raw_rows)
    TAB.mkdir(parents=True, exist_ok=True)
    RELEASE.mkdir(parents=True, exist_ok=True)
    raw.to_csv(TAB / "spatial_x_corrupt_raw.csv", index=False)
    raw.to_csv(RELEASE / "spatial_x_corrupt_raw.csv", index=False)

    # Family-mean board for locked blend + CE
    board_rows = []
    for method in ["blend_locked_0.3rob+0.7erm", "baseline_ce_seedavg"]:
        for fold_name in folds:
            for family in eval_families:
                sub = raw[
                    (raw.method == method)
                    & (raw.fold == fold_name)
                    & (raw.corruption == family)
                ]
                if family == "clean":
                    mae = float(sub["mae_stage_macro"].iloc[0])
                    board_rows.append(
                        dict(
                            stem=method,
                            fold=fold_name,
                            corruption="clean",
                            severity="family_mean",
                            mae_stage_macro=mae,
                            n_levels=1,
                            crossed_region_x_corrupt=True,
                        )
                    )
                else:
                    mae = float(sub["mae_stage_macro"].mean())
                    board_rows.append(
                        dict(
                            stem=method,
                            fold=fold_name,
                            corruption=family,
                            severity="family_mean",
                            mae_stage_macro=mae,
                            n_levels=int(len(sub)),
                            crossed_region_x_corrupt=True,
                        )
                    )
                    for _, r in sub.iterrows():
                        board_rows.append(
                            dict(
                                stem=method,
                                fold=fold_name,
                                corruption=family,
                                severity=r["severity"],
                                mae_stage_macro=float(r["mae_stage_macro"]),
                                n_levels=1,
                                crossed_region_x_corrupt=True,
                            )
                        )

    board = pd.DataFrame(board_rows)
    board.to_csv(TAB / "failure_map_crossed_board.csv", index=False)
    board.to_csv(RELEASE / "failure_map_crossed_board.csv", index=False)

    # Compact 2×3 cells for locked stem (clean / bright / noise × A2B / B2A)
    alias = {"gauss_noise_sigma": "noise", "brightness": "brightness", "clean": "clean"}
    cells = []
    for fold_name, fold_short in [
        ("spatial_A_to_B", "A_to_B"),
        ("spatial_B_to_A", "B_to_A"),
    ]:
        for fam, short in alias.items():
            sub = board[
                (board.stem == "blend_locked_0.3rob+0.7erm")
                & (board.fold == fold_name)
                & (board.corruption == fam)
                & (board.severity == "family_mean")
            ]
            cells.append(
                {
                    "stem": "Audit-OrdNet (locked)",
                    "region": fold_short,
                    "corruption": short,
                    "mae_stage_macro": float(sub["mae_stage_macro"].iloc[0]),
                    "crossed_region_x_corrupt": True,
                }
            )
            # also CE
            sub_ce = board[
                (board.stem == "baseline_ce_seedavg")
                & (board.fold == fold_name)
                & (board.corruption == fam)
                & (board.severity == "family_mean")
            ]
            cells.append(
                {
                    "stem": "ResNet-18 CE",
                    "region": fold_short,
                    "corruption": short,
                    "mae_stage_macro": float(sub_ce["mae_stage_macro"].iloc[0]),
                    "crossed_region_x_corrupt": True,
                }
            )
    cells_df = pd.DataFrame(cells)
    cells_df.to_csv(TAB / "failure_map_crossed_cells.csv", index=False)
    cells_df.to_csv(RELEASE / "failure_map_crossed_cells.csv", index=False)

    summary = {
        "stem_primary": "Audit-OrdNet (locked)",
        "blend": "0.3*aord_rob + 0.7*aord_erm seedavg",
        "also_scored": "ResNet-18 CE seedavg",
        "regions": ["spatial_A_to_B", "spatial_B_to_A"],
        "corruptions": list(eval_families.keys()),
        "true_crossed_region_x_corrupt_available": True,
        "n_raw_rows": int(len(raw)),
        "elapsed_seconds": round(time.time() - t0, 1),
        "cells_locked": {
            f"{r['region']}__{r['corruption']}": round(r["mae_stage_macro"], 4)
            for _, r in cells_df[cells_df.stem == "Audit-OrdNet (locked)"].iterrows()
        },
        "cells_ce": {
            f"{r['region']}__{r['corruption']}": round(r["mae_stage_macro"], 4)
            for _, r in cells_df[cells_df.stem == "ResNet-18 CE"].iterrows()
        },
    }
    (TAB / "failure_map_crossed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RELEASE / "failure_map_crossed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary, indent=2))
    print("CELLS\n", cells_df.to_string(index=False))
    print("ELAPSED", summary["elapsed_seconds"])


if __name__ == "__main__":
    main()
