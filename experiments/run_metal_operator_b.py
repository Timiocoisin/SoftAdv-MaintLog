#!/usr/bin/env python3
"""B-upgrade: MetalField structural operator (not 4th-channel concat).

Trains constrained metal_op (luminance τ gate + RGB modulation) versus
unconstrained free_op (1x1 spatial gate) under Leak-Guard-10, same OrdProg
dual-expert recipe. Reports necessity / auditability board.

Usage:
  PY=/root/anaconda3/envs/apds/bin/python
  $PY run_metal_operator_b.py all
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
TAB = OUT / "tables"
FIG_DIR = (
    ROOT
    / "elsarticle/figures"
)
RELEASE = (
    ROOT
    / "elsarticle/release_tables"
)
sys.path.insert(0, str(ROOT))

from training.ordprog_net import (  # noqa: E402
    VARIANTS,
    OrdProgConfig,
    OrdProgNet,
    ordprog_loss,
)

SEEDS = [0, 1, 2]
MAX_EPOCHS = 16
IMAGENET_CKPT = Path("/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
FOLD = "multi_gap_guard10"
OPERATOR_VARIANTS = [
    "ordprog_metal_op",
    "appearance_metal_op",
    "ordprog_metal_op_learn",
    "appearance_metal_op_learn",
    "ordprog_free_op",
    "appearance_free_op",
]


def stage_macro_mae(y, p):
    return float(pd.DataFrame({"g": y, "e": np.abs(y - p)}).groupby("g").e.mean().mean())


def load_pass_arrays():
    a = np.load(OUT / "sampled_data.npz")
    return a["stages"], a["keys"]


def _train_jobs(job_list, worker_id):
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

    def make_net(cfg: OrdProgConfig, seed: int):
        seed_all(seed)
        model = OrdProgNet(cfg, imagenet_state=weight).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        return model, opt, sch

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
        return total / max(len(ids), 1)

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
    print(f"METAL_OP_WORKER{worker_id}_JOBS", len(my_jobs), "/", len(job_list), flush=True)
    for cfg, seed, tag in my_jobs:
        result_path = OUT / f"{tag}.json"
        if result_path.exists() and (OUT / f"{tag}_predictions.csv").exists():
            print("SKIP", tag, flush=True)
            continue
        fit = np.isin(g, fold["inner_fit_stages"])
        val = np.isin(g, fold["inner_val_stages"])
        train = fit | val
        test = np.isin(g, fold["test_stages"])
        core = np.isin(g, fold["core_test_stages"])
        fitids = torch.tensor(np.flatnonzero(np.repeat(fit, 3)), device=device)
        trainids = torch.tensor(np.flatnonzero(np.repeat(train, 3)), device=device)

        start = time.time()
        model, opt, sch = make_net(cfg, seed)
        best = (float("inf"), None)
        for epoch in range(1, MAX_EPOCHS + 1):
            train_epoch(model, opt, cfg, fitids)
            sch.step()
            ph = predict_pass(model, val)
            vm = stage_macro_mae(g[val], ph)
            if vm < best[0]:
                best = (vm, epoch)
        del model, opt, sch
        torch.cuda.empty_cache()

        model, opt, sch = make_net(cfg, seed)
        for epoch in range(1, best[1] + 1):
            train_epoch(model, opt, cfg, trainids)
            sch.step()
        ph = predict_pass(model, test)
        core_m = core[test]
        learned_tau = model.learned_tau()
        ckpt = OUT / f"{tag}.pt"
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "cfg": cfg.to_dict(),
                "fold": FOLD,
                "seed": seed,
                "epochs": best[1],
                "learned_tau": learned_tau,
            },
            ckpt,
        )
        pd.DataFrame(
            {
                "fold": FOLD,
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
            model=cfg.name,
            seed=seed,
            appearance_mode=cfg.appearance_mode,
            metal_gate_mode=cfg.metal_gate_mode,
            metal_floor=cfg.metal_floor,
            selected_epochs=best[1],
            inner_val_mae=best[0],
            mae_stage_macro_core45=stage_macro_mae(g[test][core_m], ph[core_m]),
            learned_tau=learned_tau,
            elapsed_seconds=time.time() - start,
            checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        )
        result_path.write_text(json.dumps(row, indent=2))
        print(
            "COMPLETE",
            worker_id,
            tag,
            "core45",
            round(row["mae_stage_macro_core45"], 3),
            "mode",
            cfg.appearance_mode,
            flush=True,
        )
        del model, opt, sch
        torch.cuda.empty_cache()


def train_jobs():
    jobs = []
    for vname in OPERATOR_VARIANTS:
        for seed in SEEDS:
            tag = f"{FOLD}__{vname}__seed{seed}"
            jobs.append((VARIANTS[vname], seed, tag))
    return jobs


def train_main():
    jobs = train_jobs()
    missing = [
        j
        for j in jobs
        if not ((OUT / f"{j[2]}.json").exists() and (OUT / f"{j[2]}_predictions.csv").exists())
    ]
    print("MISSING_METAL_OP_JOBS", len(missing), flush=True)
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


def _seedavg_blend(coral_name: str, ce_name: str) -> float:
    preds, stages = [], None
    for seed in SEEDS:
        pc = pd.read_csv(OUT / f"{FOLD}__{coral_name}__seed{seed}_predictions.csv")
        pa = pd.read_csv(OUT / f"{FOLD}__{ce_name}__seed{seed}_predictions.csv")
        pc = pc[pc["in_core45"]].sort_values(["stage", "pass_id"])
        pa = pa[pa["in_core45"]].sort_values(["stage", "pass_id"])
        preds.append(0.5 * (pc["prediction"].values + pa["prediction"].values))
        stages = pc["stage"].values
    return stage_macro_mae(stages, np.mean(np.stack(preds, 0), 0))


def _seedavg_one(model: str) -> float:
    preds, stages = [], None
    for s in SEEDS:
        p = pd.read_csv(OUT / f"{FOLD}__{model}__seed{s}_predictions.csv")
        p = p[p["in_core45"]].sort_values(["stage", "pass_id"])
        preds.append(p["prediction"].values)
        stages = p["stage"].values
    return stage_macro_mae(stages, np.mean(np.stack(preds, 0), 0))


def make_operator_figure():
    """Schematic: constrained MetalField vs free gate vs 4th-channel concat."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F5F5F5", ec="#333"):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=fc, edgecolor=ec, lw=1.2
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)

    # Row A: metal_op
    ax.text(0.15, 3.55, "A. MetalField operator (constrained)", fontsize=9, fontweight="bold")
    box(0.2, 2.7, 1.4, 0.7, "RGB $x$", fc="#DDEEFF")
    box(1.9, 2.7, 1.8, 0.7, r"$m=\sigma((I-\tau)/\beta)$", fc="#D5F5E3")
    box(4.0, 2.7, 2.2, 0.7, r"$\tilde x=x\odot(\lambda+(1-\lambda)m)$", fc="#FCF3CF")
    box(6.5, 2.7, 1.5, 0.7, "ResNet-18\n(3-ch)", fc="#FADBD8")
    box(8.3, 2.7, 1.4, 0.7, r"$\hat g$", fc="#E8DAEF")
    for x0, x1 in [(1.6, 1.9), (3.7, 4.0), (6.2, 6.5), (8.0, 8.3)]:
        ax.annotate("", xy=(x1, 3.05), xytext=(x0, 3.05), arrowprops=dict(arrowstyle="->", lw=1.0))

    # Row B: free_op
    ax.text(0.15, 2.15, "B. Free spatial gate (control)", fontsize=9, fontweight="bold")
    box(0.2, 1.3, 1.4, 0.7, "RGB $x$", fc="#DDEEFF")
    box(1.9, 1.3, 1.8, 0.7, r"$m=\sigma(W*I_{\mathrm{gray}})$", fc="#F5B7B1")
    box(4.0, 1.3, 2.2, 0.7, "same modulate", fc="#FCF3CF")
    box(6.5, 1.3, 1.5, 0.7, "ResNet-18\n(3-ch)", fc="#FADBD8")
    box(8.3, 1.3, 1.4, 0.7, r"$\hat g$", fc="#E8DAEF")
    for x0, x1 in [(1.6, 1.9), (3.7, 4.0), (6.2, 6.5), (8.0, 8.3)]:
        ax.annotate("", xy=(x1, 1.65), xytext=(x0, 1.65), arrowprops=dict(arrowstyle="->", lw=1.0))

    # Row C: concat ref
    ax.text(0.15, 0.75, "C. Fourth-channel concat (published stem)", fontsize=9, fontweight="bold")
    box(0.2, 0.05, 1.4, 0.55, "RGB $x$", fc="#DDEEFF")
    box(1.9, 0.05, 1.8, 0.55, r"$m=1[I>\tau]$", fc="#D5F5E3")
    box(4.0, 0.05, 2.2, 0.55, r"$[x; m]$ 4-ch", fc="#D5D8DC")
    box(6.5, 0.05, 1.5, 0.55, "ResNet-18\n(4-ch)", fc="#FADBD8")
    box(8.3, 0.05, 1.4, 0.55, r"$\hat g$", fc="#E8DAEF")
    for x0, x1 in [(1.6, 1.9), (3.7, 4.0), (6.2, 6.5), (8.0, 8.3)]:
        ax.annotate("", xy=(x1, 0.32), xytext=(x0, 0.32), arrowprops=dict(arrowstyle="->", lw=1.0))

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = "Fig15_metal_field_operator"
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600)
    plt.close(fig)
    print("FIG15", FIG_DIR / f"{stem}.pdf", flush=True)


def summarize():
    TAB.mkdir(parents=True, exist_ok=True)
    RELEASE.mkdir(parents=True, exist_ok=True)

    fixed_blend = 7.4311
    learn_concat = 7.508076740299824  # from auditable_metal_gates
    otsu_concat = 8.654085565712519

    metal_op_blend = _seedavg_blend("ordprog_metal_op", "appearance_metal_op")
    metal_op_learn_blend = _seedavg_blend("ordprog_metal_op_learn", "appearance_metal_op_learn")
    free_op_blend = _seedavg_blend("ordprog_free_op", "appearance_free_op")
    # CORAL-only readout is the stable operator score (CE metal_op seed1 collapses).
    metal_op_coral = _seedavg_one("ordprog_metal_op")
    metal_op_learn_coral = _seedavg_one("ordprog_metal_op_learn")
    free_op_coral = _seedavg_one("ordprog_free_op")

    taus = []
    for v in ("ordprog_metal_op_learn", "appearance_metal_op_learn"):
        for s in SEEDS:
            d = json.loads((OUT / f"{FOLD}__{v}__seed{s}.json").read_text())
            if d.get("learned_tau") is not None:
                taus.append(float(d["learned_tau"]))

    board = pd.DataFrame(
        [
            dict(
                fuse="4ch_concat",
                gate="fixed_tau",
                method="OrdProg blend (published)",
                mae_seedavg=fixed_blend,
                mae_coral_only=float("nan"),
                learned_tau=28 / 255,
                auditable=True,
                note="stem ranking recipe",
            ),
            dict(
                fuse="4ch_concat",
                gate="learnable_tau",
                method="OrdProg blend",
                mae_seedavg=learn_concat,
                mae_coral_only=float("nan"),
                learned_tau=float("nan"),
                auditable=True,
                note="from auditable_metal_gates",
            ),
            dict(
                fuse="4ch_concat",
                gate="otsu",
                method="OrdProg blend",
                mae_seedavg=otsu_concat,
                mae_coral_only=float("nan"),
                learned_tau=float("nan"),
                auditable=True,
                note="from auditable_metal_gates",
            ),
            dict(
                fuse="metal_op",
                gate="fixed_tau",
                method="MetalField + CORAL (primary operator)",
                mae_seedavg=metal_op_blend,
                mae_coral_only=metal_op_coral,
                learned_tau=28 / 255,
                auditable=True,
                note="primary B score uses CORAL-only; blend hurt by CE seed variance",
            ),
            dict(
                fuse="metal_op",
                gate="learnable_tau",
                method="MetalField + CORAL",
                mae_seedavg=metal_op_learn_blend,
                mae_coral_only=metal_op_learn_coral,
                learned_tau=float(np.mean(taus)) if taus else float("nan"),
                auditable=True,
                note="learnable τ + MetalField",
            ),
            dict(
                fuse="free_op",
                gate="unconstrained_1x1",
                method="free gate blend (control)",
                mae_seedavg=free_op_blend,
                mae_coral_only=free_op_coral,
                learned_tau=float("nan"),
                auditable=False,
                note="can lower MAE; rejected as claimed prior (not luminance-τ)",
            ),
        ]
    )
    # expert rows
    experts = []
    for v in OPERATOR_VARIANTS:
        experts.append(
            dict(
                model=v,
                appearance_mode=VARIANTS[v].appearance_mode,
                metal_gate_mode=VARIANTS[v].metal_gate_mode,
                mae_seedavg_pred=_seedavg_one(v),
            )
        )
    expert_df = pd.DataFrame(experts)

    # Necessity chain (reuse published refs)
    necessity = pd.DataFrame(
        [
            dict(rung="scalar_rho_to_g", mae=26.94, source="main table locked map"),
            dict(rung="mask_geometry_stagefit", mae=14.67, source="auditable_mask_geometry"),
            dict(rung="mask_only_CNN", mae=7.90, source="main table"),
            dict(rung="4ch_concat_OrdProg", mae=fixed_blend, source="main table"),
            dict(rung="metal_op_CORAL", mae=metal_op_coral, source="this run; primary operator"),
            dict(rung="metal_op_learn_CORAL", mae=metal_op_learn_coral, source="this run"),
            dict(rung="metal_op_blend", mae=metal_op_blend, source="this run; CE seed-unstable"),
            dict(rung="free_op_blend", mae=free_op_blend, source="this run; non-auditable control"),
            dict(rung="w_o_metal_fusion", mae=11.14, source="metal ablation table"),
        ]
    )

    board.to_csv(TAB / "metal_field_operator_board.csv", index=False)
    board.to_csv(RELEASE / "metal_field_operator_board.csv", index=False)
    expert_df.to_csv(TAB / "metal_field_operator_experts.csv", index=False)
    expert_df.to_csv(RELEASE / "metal_field_operator_experts.csv", index=False)
    necessity.to_csv(TAB / "metal_field_necessity_chain.csv", index=False)
    necessity.to_csv(RELEASE / "metal_field_necessity_chain.csv", index=False)

    headline = dict(
        name="MetalField structural operator (B-upgrade)",
        definition={
            "m_constrained": "m=1[I>τ] or soft σ((I-τ)/β); luminance-only",
            "operator": "x' = x ⊙ (λ + (1-λ) m), λ=0.25; 3-ch ResNet",
            "free_control": "m=σ(W*I_gray); unconstrained 1x1",
            "note": "Operator is auditable; free gate is the non-auditable control",
        },
        mae_4ch_concat_fixed=fixed_blend,
        mae_metal_op_coral=float(metal_op_coral),
        mae_metal_op_learn_coral=float(metal_op_learn_coral),
        mae_metal_op_blend=float(metal_op_blend),
        mae_free_op_blend=float(free_op_blend),
        mae_free_op_coral=float(free_op_coral),
        learned_tau_mean=float(np.mean(taus)) if taus else None,
        delta_op_coral_minus_concat=float(metal_op_coral - fixed_blend),
        delta_free_blend_minus_op_coral=float(free_op_blend - metal_op_coral),
        primary_operator_score="metal_op CORAL-only seedavg-pred",
    )
    (TAB / "metal_field_operator_headline.json").write_text(json.dumps(headline, indent=2))
    (RELEASE / "metal_field_operator_headline.json").write_text(json.dumps(headline, indent=2))

    make_operator_figure()
    print(board.to_string(index=False))
    print(necessity.to_string(index=False))
    print(json.dumps(headline, indent=2))


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "all":
        train_main()
        summarize()
        return
    cmd = sys.argv[1]
    if cmd == "train":
        if sys.argv[2] == "main":
            train_main()
        elif sys.argv[2] == "worker":
            train_worker(int(sys.argv[3]))
        else:
            raise SystemExit("train main|worker")
    elif cmd == "summarize":
        summarize()
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main()
