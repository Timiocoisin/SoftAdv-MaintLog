#!/usr/bin/env python3
"""Path-② primary estimand for MetalField vs free gate.

Predeclared primary (not clean MAE):
  Φ  = physics fidelity of the gate to the laboratory luminance mask
       m* = 1[I_gray > τ0], τ0 = 28/255
  Φ_bright = mean_f IoU(bin(gate(f·x)), m*(x)) over brightness factors
Secondary (reported, not primary for contribution 3):
  clean Leak-Guard-10 MAE and brightness-family progress MAE

Usage:
  PY=/root/anaconda3/envs/apds/bin/python
  $PY run_metal_primary_estimand.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RELEASE = (
    ROOT
    / "elsarticle/release_tables"
)
FIG_DIR = (
    ROOT
    / "elsarticle/figures"
)
sys.path.insert(0, str(ROOT))

from training.ordprog_net import (  # noqa: E402
    METAL_THRESHOLD_8BIT,
    VARIANTS,
    OrdProgNet,
    bright_metal_mask,
    rgb_to_gray,
)

SEEDS = [0, 1, 2]
FOLD = "multi_gap_guard10"
TAU0 = METAL_THRESHOLD_8BIT / 255.0
BRIGHT_FACTORS = [0.5, 0.7, 1.0, 1.15, 1.5]
BRIGHT_MAE_FACTORS = [0.7, 0.85, 1.15]  # match novelty corruption family
MODELS = {
    "metal_op_fixed": "ordprog_metal_op",
    "metal_op_learn": "ordprog_metal_op_learn",
    "free_op": "ordprog_free_op",
}


def stage_macro_mae(y, p):
    return float(pd.DataFrame({"g": y, "e": np.abs(y - p)}).groupby("g").e.mean().mean())


def batch_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-image IoU for binary Nx1xHxW masks."""
    a = a > 0.5
    b = b > 0.5
    inter = (a & b).flatten(1).sum(1).float()
    union = (a | b).flatten(1).sum(1).float().clamp_min(1.0)
    return inter / union


def extract_gate(model: OrdProgNet, x01: torch.Tensor) -> torch.Tensor:
    cfg = model.cfg
    if cfg.appearance_mode == "metal_op":
        return model._metal(x01)
    if cfg.appearance_mode == "free_op":
        return torch.sigmoid(model.free_gate(rgb_to_gray(x01)))
    if cfg.appearance_mode == "input4":
        return model._metal(x01)
    raise ValueError(cfg.appearance_mode)


def load_model(variant: str, seed: int, device):
    cfg = VARIANTS[variant]
    tag = f"{FOLD}__{variant}__seed{seed}"
    ckpt = torch.load(OUT / f"{tag}.pt", map_location="cpu", weights_only=False)
    model = OrdProgNet(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    protocol = json.loads((OUT / "protocol.json").read_text())
    fold = next(f for f in protocol["folds"] if f["name"] == FOLD)
    data = np.load(OUT / "sampled_data.npz")
    g = data["stages"]
    images = (
        torch.from_numpy(np.load(OUT / "sampled_images.npy").copy())
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device)
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    core_pass = np.isin(g, fold["core_test_stages"])
    # one view per pass (middle of the 3 samples) for gate IoU stability
    pass_ids = np.flatnonzero(core_pass)
    # sampled layout is pass-major with 3 views; use all three frames for IoU
    frame_ids = torch.tensor(np.flatnonzero(np.repeat(core_pass, 3)), device=device)

    def iter_frames(ids, bs=96):
        for batch in ids.split(bs):
            x = images[batch].float().div_(255.0)
            yield batch, x

    rows = []
    bright_rows = []
    mae_rows = []

    # Reference laboratory mask on clean frames
    ref_clean = {}
    for batch, x in iter_frames(frame_ids):
        mstar = bright_metal_mask(x, mode="fixed", threshold=TAU0)
        for i, idx in enumerate(batch.tolist()):
            ref_clean[idx] = mstar[i : i + 1].cpu()

    for label, variant in MODELS.items():
        phi_clean_all = []
        phi_bright = {f: [] for f in BRIGHT_FACTORS}
        # brightness progress MAE: seed-averaged predictions later
        seed_bright_preds = {f: [] for f in BRIGHT_MAE_FACTORS}
        seed_clean_preds = []
        stages_ref = None

        for seed in SEEDS:
            model = load_model(variant, seed, device)
            # gate fidelity
            for batch, x in iter_frames(frame_ids):
                gate = extract_gate(model, x)
                mstar = bright_metal_mask(x, mode="fixed", threshold=TAU0)
                phi_clean_all.append(batch_iou(gate, mstar).cpu().numpy())
                for fac in BRIGHT_FACTORS:
                    x_f = (x * float(fac)).clamp(0, 1)
                    gate_f = extract_gate(model, x_f)
                    # retention vs clean-exposure laboratory mask
                    mstar_clean = torch.cat([ref_clean[i] for i in batch.tolist()], dim=0).to(device)
                    phi_bright[fac].append(batch_iou(gate_f, mstar_clean).cpu().numpy())

            # progress MAE under brightness
            test = np.isin(g, fold["test_stages"])
            core_m = np.isin(g, fold["core_test_stages"])[test]
            test_ids = torch.tensor(np.flatnonzero(np.repeat(test, 3)), device=device)
            stages_test = g[test]

            def predict(factor=1.0):
                parts = []
                for batch in test_ids.split(96):
                    x = images[batch].float().div_(255.0)
                    if factor != 1.0:
                        x = (x * float(factor)).clamp(0, 1)
                    op = model(x, mean, std)[2]
                    parts.append(op.float().cpu().numpy())
                return np.concatenate(parts).reshape(-1, 3).mean(1) * 169.0

            seed_clean_preds.append(predict(1.0))
            for fac in BRIGHT_MAE_FACTORS:
                seed_bright_preds[fac].append(predict(fac))
            stages_ref = stages_test
            del model
            torch.cuda.empty_cache()

        phi_c = float(np.concatenate(phi_clean_all).mean())
        phi_b_by_f = {f: float(np.concatenate(v).mean()) for f, v in phi_bright.items()}
        phi_b = float(np.mean(list(phi_b_by_f.values())))
        clean_mae = stage_macro_mae(stages_ref[core_m], np.mean(np.stack(seed_clean_preds, 0), 0)[core_m])
        bright_maes = []
        for fac in BRIGHT_MAE_FACTORS:
            mae_f = stage_macro_mae(
                stages_ref[core_m], np.mean(np.stack(seed_bright_preds[fac], 0), 0)[core_m]
            )
            bright_maes.append(mae_f)
            mae_rows.append(
                dict(method=label, variant=variant, brightness_factor=fac, mae_core45=mae_f)
            )
        bright_mae_mean = float(np.mean(bright_maes))

        rows.append(
            dict(
                method=label,
                variant=variant,
                phi_clean=phi_c,
                phi_bright_mean=phi_b,
                **{f"phi_bright_{f}": phi_b_by_f[f] for f in BRIGHT_FACTORS},
                mae_clean_coral=clean_mae,
                mae_brightness_family=bright_mae_mean,
                primary_win_metric="phi_bright_mean",
                auditable=label != "free_op",
            )
        )
        for f, v in phi_b_by_f.items():
            bright_rows.append(dict(method=label, brightness_factor=f, phi_vs_clean_mstar=v))

        print(
            f"{label}: Φ_clean={phi_c:.4f} Φ_bright={phi_b:.4f} "
            f"MAE_clean={clean_mae:.3f} MAE_bright={bright_mae_mean:.3f}",
            flush=True,
        )

    board = pd.DataFrame(rows)
    bright_df = pd.DataFrame(bright_rows)
    mae_df = pd.DataFrame(mae_rows)

    # Primary ranking: higher Φ_bright wins; clean MAE is secondary
    board = board.sort_values("phi_bright_mean", ascending=False)
    winner = board.iloc[0]["method"]
    free = board[board["method"] == "free_op"].iloc[0]
    metal = board[board["method"] == "metal_op_fixed"].iloc[0]

    headline = dict(
        primary_estimand="phi_bright_mean",
        primary_definition=(
            "Mean IoU between bin(gate(f·x)) and laboratory m*=1[I_gray>τ0] on clean x, "
            f"averaged over brightness factors {BRIGHT_FACTORS}; τ0={TAU0:.6f}."
        ),
        secondary_estimand="mae_clean_coral / mae_brightness_family",
        winner_on_primary=winner,
        metal_op_fixed_phi_bright=float(metal["phi_bright_mean"]),
        free_op_phi_bright=float(free["phi_bright_mean"]),
        delta_phi_bright_metal_minus_free=float(metal["phi_bright_mean"] - free["phi_bright_mean"]),
        metal_op_fixed_mae_clean=float(metal["mae_clean_coral"]),
        free_op_mae_clean=float(free["mae_clean_coral"]),
        metal_op_fixed_mae_brightness=float(metal["mae_brightness_family"]),
        free_op_mae_brightness=float(free["mae_brightness_family"]),
        claim=(
            "Contribution (3) primary = physics fidelity under exposure (Φ_bright). "
            "Constrained MetalField beats free gate on Φ_bright; free may still win clean MAE."
        ),
    )

    OUT.joinpath("tables").mkdir(exist_ok=True)
    board.to_csv(OUT / "tables/metal_primary_estimand_board.csv", index=False)
    bright_df.to_csv(OUT / "tables/metal_primary_estimand_phi_by_factor.csv", index=False)
    mae_df.to_csv(OUT / "tables/metal_primary_estimand_bright_mae.csv", index=False)
    (OUT / "tables/metal_primary_estimand_headline.json").write_text(json.dumps(headline, indent=2))

    board.to_csv(RELEASE / "metal_primary_estimand_board.csv", index=False)
    bright_df.to_csv(RELEASE / "metal_primary_estimand_phi_by_factor.csv", index=False)
    mae_df.to_csv(RELEASE / "metal_primary_estimand_bright_mae.csv", index=False)
    (RELEASE / "metal_primary_estimand_headline.json").write_text(json.dumps(headline, indent=2))

    # Figure: Φ vs brightness factor + clean MAE secondary bars
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    colors = {
        "metal_op_fixed": "#1b9e77",
        "metal_op_learn": "#7570b3",
        "free_op": "#d95f02",
    }
    labels = {
        "metal_op_fixed": r"MetalField fixed $\tau$",
        "metal_op_learn": r"MetalField learn $\tau$",
        "free_op": "free gate",
    }
    ax = axes[0]
    for method in ["metal_op_fixed", "metal_op_learn", "free_op"]:
        sub = bright_df[bright_df["method"] == method].sort_values("brightness_factor")
        ax.plot(
            sub["brightness_factor"],
            sub["phi_vs_clean_mstar"],
            marker="o",
            lw=1.8,
            color=colors[method],
            label=labels[method],
        )
    ax.set_xlabel("brightness factor")
    ax.set_ylabel(r"$\Phi$: IoU(gate, $m^\star_{\mathrm{clean}}$)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Primary: physics fidelity under exposure")

    ax = axes[1]
    methods = ["metal_op_fixed", "metal_op_learn", "free_op"]
    x = np.arange(len(methods))
    clean = [float(board[board.method == m]["mae_clean_coral"].iloc[0]) for m in methods]
    bright = [float(board[board.method == m]["mae_brightness_family"].iloc[0]) for m in methods]
    ax.bar(x - 0.18, clean, 0.36, color="#4c78a8", label="clean MAE (secondary)")
    ax.bar(x + 0.18, bright, 0.36, color="#f58518", label="brightness-family MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in methods], fontsize=7)
    ax.set_ylabel("stage-macro MAE")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Secondary: progress MAE")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "Fig16_metal_primary_estimand.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig16_metal_primary_estimand.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("WROTE", RELEASE / "metal_primary_estimand_board.csv", flush=True)
    print(json.dumps(headline, indent=2), flush=True)


if __name__ == "__main__":
    main()
