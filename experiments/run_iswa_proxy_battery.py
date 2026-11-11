#!/usr/bin/env python3
"""ISWA proxy battery: heuristics, self-consistency, camera curve, leave-band,
campaign/pass splits, failure-case board. Uses apds CUDA for new CNN folds.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/iet_resubmit_20261106"
FAIR = ROOT / "outputs/fair_cnn_validation_bf16_20261106"
TAB = ROOT / "elsarticle/release_tables"
FIG = ROOT / "elsarticle/figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

INK, MUTE, RULE = "#111111", "#555555", "#222222"
OK, BAD, SKY, GREY = "#009E73", "#D55E00", "#0072B2", "#7A7A7A"
GMAX = 169
H = 4
CORE = {25, 55, 85, 115, 145}
CORE_STAGES = sorted(
    {g for c in CORE for g in range(c - H, c + H + 1) if 1 <= g <= GMAX}
)
SEEDS = [0, 1, 2]
MAX_EPOCHS = 16
BATCH = 48
IMAGENET_CKPT = Path("/root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
PYTORCH = sys.executable


def style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8.0,
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": RULE,
            "pdf.fonttype": 42,
        }
    )


def stage_macro_mae(y, p):
    return float(pd.DataFrame({"g": y, "e": np.abs(y - p)}).groupby("g")["e"].mean().mean())


def leak10_train_stages():
    proto = json.loads(
        (ROOT / "elsarticle/release_tables_src/leak_guard_splits/leak_guard_protocol.json").read_text()
    )
    fold = next(f for f in proto["folds"] if f["guard"] == 10)
    return set(int(s) for s in fold["train_stages"]), set(int(s) for s in fold["core_test_stages"])


def load_pass_feat():
    df = pd.read_csv(ROOT / "outputs/idea_validation_20261106/pass_features.csv")
    return df


def load_seedavg_preds(pattern: str, seeds=(0, 1, 2)):
    bags = defaultdict(list)
    meta = {}
    for seed in seeds:
        path = next(OUT.glob(pattern.format(seed=seed)))
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = (int(row["stage"]), int(row["pass_id"]))
            bags[key].append(float(row["prediction"]))
            meta[key] = int(row["stage"])
    rows = []
    for key, vals in bags.items():
        rows.append(
            dict(
                stage=meta[key],
                pass_id=key[1],
                ghat=float(np.mean(vals)),
                in_core=meta[key] in CORE_STAGES,
            )
        )
    return pd.DataFrame(rows)


# ---------- heuristics ----------
def run_heuristics():
    df = load_pass_feat()
    train_st, core_st = leak10_train_stages()
    # pass-mean bright_40 per stage on train
    tr = df[df.stage.isin(train_st)]
    te = df[df.stage.isin(core_st)]
    stage_rho = tr.groupby("stage")["bright_40"].mean()
    stage_g = stage_rho.index.to_numpy(dtype=float)
    rho = stage_rho.to_numpy(dtype=float)

    iso = IsotonicRegression(out_of_bounds="clip").fit(rho, stage_g)
    # evaluate per core pass then stage-macro
    def eval_map(fn):
        pred = te["bright_40"].to_numpy()
        ghat = fn(pred)
        return stage_macro_mae(te["stage"].to_numpy(), ghat)

    mae_iso = eval_map(lambda r: iso.predict(r))

    # fixed equal-width ρ bins → mean train stage
    edges = np.linspace(0, max(1e-6, float(tr["bright_40"].max())), 11)
    dig = np.digitize(tr["bright_40"], edges[1:-1], right=True)
    bin_mean = (
        pd.DataFrame({"b": dig, "g": tr["stage"]}).groupby("b")["g"].mean().to_dict()
    )
    fallback = float(tr["stage"].mean())

    def bin_predict(r):
        b = np.digitize(r, edges[1:-1], right=True)
        return np.array([bin_mean.get(int(x), fallback) for x in b], dtype=float)

    mae_bin = eval_map(bin_predict)

    # piecewise linear on stage-mean ρ (3 segments by ρ tertiles)
    order = np.argsort(rho)
    rho_s, g_s = rho[order], stage_g[order]
    n = len(rho_s)
    cuts = [0, n // 3, 2 * n // 3, n]
    segs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 2:
            continue
        lr = LinearRegression().fit(rho_s[a:b].reshape(-1, 1), g_s[a:b])
        segs.append((float(rho_s[a]), float(rho_s[b - 1]) if b < n else 1.0, lr))

    def pwl_predict(r):
        out = np.empty_like(r, dtype=float)
        for i, x in enumerate(r):
            hit = segs[-1][2]
            for lo, hi, lr in segs:
                if x <= hi or lr is segs[-1][2]:
                    hit = lr
                    if x <= hi:
                        break
            out[i] = float(hit.predict(np.array([[x]]))[0])
        return np.clip(out, 1, GMAX)

    mae_pwl = eval_map(pwl_predict)

    # Audit / CE seedavg stage-macro on core for reference
    audit = load_seedavg_preds("auditord__multi_gap_guard10__aord_erm__seed{seed}_predictions.csv")
    ce = load_seedavg_preds("multi_gap_guard10__baseline_ce__seed{seed}_predictions.csv")
    a_core = audit[audit.in_core]
    c_core = ce[ce.in_core]
    out = {
        "description": "Non-learning appearance heuristics on bright_40; hard stage-gap hold ($k^*=10$) train→core. Not a multi-rater human study.",
        "isotonic_mae_core": mae_iso,
        "rho_bin10_mae_core": mae_bin,
        "piecewise_linear_mae_core": mae_pwl,
        "audit_erm_seedavg_mae_core": stage_macro_mae(a_core.stage, a_core.ghat),
        "ce_seedavg_mae_core": stage_macro_mae(c_core.stage, c_core.ghat),
    }
    (TAB / "visual_heuristic_family.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(
        [
            {"method": "ρ isotonic", "mae_core": mae_iso},
            {"method": "ρ 10-bin lookup", "mae_core": mae_bin},
            {"method": "ρ piecewise-linear", "mae_core": mae_pwl},
            {"method": "Audit-OrdNet ERM", "mae_core": out["audit_erm_seedavg_mae_core"]},
            {"method": "ResNet-18 CE", "mae_core": out["ce_seedavg_mae_core"]},
        ]
    ).to_csv(TAB / "visual_heuristic_family.csv", index=False)
    # refresh single-heuristic json used earlier
    (TAB / "visual_heuristic_baseline.json").write_text(
        json.dumps(
            {
                "name": "bright_metal_isotonic_heuristic",
                "description": out["description"],
                "mae_core": mae_iso,
                "rho_bin10_mae_core": mae_bin,
                "piecewise_linear_mae_core": mae_pwl,
                "audit_mae_core_stage": out["audit_erm_seedavg_mae_core"],
                "ce_mae_core_stage": out["ce_seedavg_mae_core"],
            },
            indent=2,
        )
    )
    print("HEURISTICS", out)
    return out


# ---------- self-consistency ----------
def run_self_consistency():
    audit = load_seedavg_preds("auditord__multi_gap_guard10__aord_erm__seed{seed}_predictions.csv")
    core = audit[audit.in_core].copy()
    # within-stage pass dispersion
    grp = core.groupby("stage")["ghat"]
    within_std = float(grp.std().mean())
    within_range = float((grp.max() - grp.min()).mean())
    # adjacent pass |Δĝ| within stage
    deltas = []
    for _, sub in core.sort_values(["stage", "pass_id"]).groupby("stage"):
        ghat = sub["ghat"].to_numpy()
        if len(ghat) >= 2:
            deltas.extend(np.abs(np.diff(ghat)).tolist())
    adj = float(np.mean(deltas)) if deltas else float("nan")
    # EMA
    rows = []
    for stage, sub in core.sort_values(["stage", "pass_id"]).groupby("stage"):
        ghat = sub["ghat"].to_numpy()
        g = float(stage)
        ema = None
        for v in ghat:
            ema = v if ema is None else 0.5 * v + 0.5 * ema
            rows.append(abs(ema - g))
        for v in ghat:
            rows.append(None)  # placeholder unused
    # cleaner EMA mae
    ema_err, raw_err = [], []
    for stage, sub in core.sort_values(["stage", "pass_id"]).groupby("stage"):
        ghat = sub["ghat"].to_numpy()
        g = float(stage)
        ema = None
        for v in ghat:
            raw_err.append(abs(v - g))
            ema = v if ema is None else 0.5 * v + 0.5 * ema
            ema_err.append(abs(ema - g))
    # stage-macro from pass-mean seedavg (existing)
    pass_mean = core.groupby("stage")["ghat"].mean()
    mae_raw = stage_macro_mae(pass_mean.index.to_numpy(), pass_mean.to_numpy())
    # rebuild EMA stage means
    ema_stage = {}
    for stage, sub in core.sort_values(["stage", "pass_id"]).groupby("stage"):
        ema = None
        vals = []
        for v in sub["ghat"].to_numpy():
            ema = v if ema is None else 0.5 * v + 0.5 * ema
            vals.append(ema)
        ema_stage[int(stage)] = float(np.mean(vals))
    mae_ema = stage_macro_mae(
        np.array(list(ema_stage.keys())), np.array(list(ema_stage.values()))
    )
    out = {
        "within_stage_pass_mean_abs_delta_ghat": adj,
        "within_stage_pass_mean_std_ghat": within_std,
        "within_stage_pass_mean_range_ghat": within_range,
        "core_mae_raw_seedavg_passmean": mae_raw,
        "core_mae_ema05": mae_ema,
        "delta_mae_ema_minus_raw": mae_ema - mae_raw,
        "n_core_stages": int(core.stage.nunique()),
        "n_core_passes": int(len(core)),
    }
    (TAB / "temporal_consistency.json").write_text(json.dumps(out, indent=2))
    print("SELF_CONSISTENCY", out)
    return out


# ---------- camera install curve (existing severity + optional affine) ----------
def run_camera_curve():
    raw = pd.read_csv(OUT / "tables/audit_ord_opt_corrupt_raw.csv")
    nov = pd.read_csv(OUT / "tables/novelty_corrupt_g10_raw.csv")
    # aord_erm severities
    a = raw[raw.method == "aord_erm"].copy()
    a["stem"] = "Audit-OrdNet ERM"
    # CE from novelty
    c = nov[nov.method == "ResNet-18 CE"].copy()
    c = c.rename(columns={"mae_stage_macro_core45": "mae"})
    c["stem"] = "ResNet-18 CE"
    a = a.rename(columns={"mae_stage_macro_core45": "mae"})
    # families of interest
    fams = ["blur_sigma", "jpeg_quality", "center_crop_frac"]
    rows = []
    for fam in fams:
        for stem_df, stem in [(a, "Audit-OrdNet ERM"), (c, "ResNet-18 CE")]:
            sub = stem_df[stem_df.corruption == fam].sort_values("severity")
            for _, r in sub.iterrows():
                rows.append(
                    dict(
                        family=fam,
                        severity=float(r["severity"]),
                        stem=stem,
                        mae=float(r["mae"]),
                    )
                )
            # clean reference
            if stem == "Audit-OrdNet ERM":
                clean = float(a.loc[a.corruption == "clean", "mae"].iloc[0])
            else:
                clean = float(c.loc[c.corruption == "clean", "mae"].iloc[0])
            rows.append(dict(family=fam, severity=np.nan, stem=stem, mae=clean, ref="clean"))
    curve = pd.DataFrame(rows)
    curve.to_csv(TAB / "camera_install_proxy_curve.csv", index=False)

    style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
    titles = {
        "blur_sigma": "Defocus (σ)",
        "jpeg_quality": "JPEG quality",
        "center_crop_frac": "Center crop frac",
    }
    for ax, fam in zip(axes, fams):
        for stem, color in [("Audit-OrdNet ERM", SKY), ("ResNet-18 CE", BAD)]:
            sub = curve[(curve.family == fam) & (curve.stem == stem) & curve.severity.notna()]
            clean = curve[
                (curve.family == fam) & (curve.stem == stem) & curve.severity.isna()
            ]["mae"].iloc[0]
            x = sub["severity"].to_numpy()
            y = sub["mae"].to_numpy()
            if fam == "jpeg_quality":
                # lower quality = harder; plot as-is
                ax.plot(x, y, "-o", color=color, ms=3.5, lw=1.2, label=stem)
            else:
                ax.plot(x, y, "-o", color=color, ms=3.5, lw=1.2, label=stem)
            ax.axhline(clean, color=color, ls=":", lw=0.8, alpha=0.7)
        ax.set_title(titles[fam], color=INK)
        ax.set_xlabel("severity")
        ax.grid(True, axis="y", color="#EEE", lw=0.6)
    axes[0].set_ylabel("Core stage-macro MAE")
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.suptitle(
        "Camera-install proxies on $k^*=10$ hold (dotted = clean)",
        fontsize=8.5,
        color=INK,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG / "Fig29_camera_install_proxy.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Fig29_camera_install_proxy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("CAMERA_CURVE", FIG / "Fig29_camera_install_proxy.pdf")
    return curve


# ---------- leave-one-band from existing fair CNN + new CNN folds ----------
def summarize_leave_band():
    rows = []
    for f in sorted(FAIR.glob("block_c*_guard10_seed*.json")):
        d = json.loads(f.read_text())
        rows.append(
            dict(
                center=int(d["center"]),
                seed=int(d["seed"]),
                mae=float(d["mae_stage_macro"]),
                model="ResNet-18 (fair CNN)",
                protocol="leave-band guard10",
            )
        )
    df = pd.DataFrame(rows)
    summ = (
        df.groupby("center")["mae"].agg(["mean", "std", "count"]).reset_index()
        if len(df)
        else pd.DataFrame()
    )
    df.to_csv(TAB / "leave_band_guard10_raw.csv", index=False)
    summ.to_csv(TAB / "leave_band_guard10_summary.csv", index=False)
    print("LEAVE_BAND_EXISTING\n", summ)
    return summ


# ---------- train campaign/pass/leave folds ----------
def build_proxy_folds(g: np.ndarray, keys: np.ndarray):
    folds = {}
    # leave mid band 115±h with guard 10
    c = 115
    test = (g >= c - H) & (g <= c + H)
    train = (g < c - H - 10) | (g > c + H + 10)
    folds["leave_band_c115_g10"] = dict(train=train, test=test, kind="leave_band")

    # campaign half
    folds["campaign_early_to_late"] = dict(
        train=g <= 84, test=g >= 85, kind="campaign_half"
    )
    folds["campaign_late_to_early"] = dict(
        train=g >= 85, test=g <= 84, kind="campaign_half"
    )

    # pass half within stage
    train_m = np.zeros(len(g), dtype=bool)
    test_m = np.zeros(len(g), dtype=bool)
    for stage in np.unique(g):
        idx = np.flatnonzero(g == stage)
        pids = keys[idx, 1]
        order = np.argsort(pids)
        idx = idx[order]
        mid = max(1, len(idx) // 2)
        if len(idx) < 2:
            train_m[idx] = True
            continue
        train_m[idx[:mid]] = True
        test_m[idx[mid:]] = True
    folds["pass_half_within_stage"] = dict(
        train=train_m, test=test_m, kind="pass_half"
    )
    return folds


def train_proxy_folds():
    import torch
    from training.ordprog_net import VARIANTS, build_model, decode_predictions, ordprog_loss

    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    g, keys = np.load(OUT / "sampled_data.npz")["stages"], np.load(OUT / "sampled_data.npz")["keys"]
    images = (
        torch.from_numpy(np.load(OUT / "sampled_images.npy").copy())
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(device)
    )
    gp = np.repeat(g, 3)
    targets = torch.tensor(gp / float(GMAX), dtype=torch.float32, device=device)
    c3 = torch.tensor(
        np.where(gp <= 40, 0, np.where(gp <= 110, 1, 2)), device=device, dtype=torch.long
    )
    c5 = torch.tensor(np.minimum((gp - 1) // 34, 4), device=device, dtype=torch.long)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    weight = torch.load(IMAGENET_CKPT, map_location="cpu", weights_only=True)
    folds = build_proxy_folds(g, keys)

    methods = {
        "baseline_ce": "baseline_ce",
        "aord_erm": "audit_ord_coral",
    }

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def batch_x(ids, augment=False):
        x = images[ids].float().div_(255.0)
        if augment:
            flip = torch.rand(len(ids), device=device) < 0.5
            x[flip] = x[flip].flip(-1)
        return x

    def pass_ids_from_mask(mask):
        return np.flatnonzero(mask)

    def frame_ids(pass_indices):
        return torch.tensor(
            np.concatenate([np.arange(p * 3, p * 3 + 3) for p in pass_indices]),
            device=device,
        )

    @torch.inference_mode()
    def predict(model, cfg, pass_indices):
        ids = frame_ids(pass_indices)
        model.eval()
        parts = []
        for batch in ids.split(96):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o3, o5, op = model(batch_x(batch), mean, std)
            dec = decode_predictions(o3.float(), o5.float(), op.float(), cfg)
            parts.append(dec["progress"].cpu().numpy())
        prog = np.concatenate(parts).reshape(-1, 3).mean(1) * GMAX
        return prog

    def train_one(variant_name, seed, train_mask, test_mask, tag):
        result_path = OUT / f"proxy__{tag}__{variant_name}__seed{seed}.json"
        if result_path.exists():
            print("SKIP", result_path.name, flush=True)
            return json.loads(result_path.read_text())
        cfg = VARIANTS[variant_name]
        train_passes = pass_ids_from_mask(train_mask)
        test_passes = pass_ids_from_mask(test_mask)
        train_stages = np.unique(g[train_passes])
        val_stages = train_stages[np.arange(len(train_stages)) % 5 == 2]
        if len(val_stages) < 3:
            val_stages = train_stages[:: max(1, len(train_stages) // 5)]
        fit_stages = np.setdiff1d(train_stages, val_stages)
        fit_passes = train_passes[np.isin(g[train_passes], fit_stages)]
        val_passes = train_passes[np.isin(g[train_passes], val_stages)]
        if len(fit_passes) < 5 or len(val_passes) < 2 or len(test_passes) < 2:
            out = dict(tag=tag, method=variant_name, seed=seed, error="too_few_passes")
            result_path.write_text(json.dumps(out, indent=2))
            return out

        seed_all(seed)
        model = build_model(cfg, imagenet_state=weight).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        fitids = frame_ids(fit_passes)
        best = (float("inf"), 1)
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            order = fitids[torch.randperm(len(fitids), device=device)]
            for batch in order.split(BATCH):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    o3, o5, op = model(batch_x(batch, True), mean, std)
                    loss = ordprog_loss(
                        o3, o5, op, c3[batch], c5[batch], targets[batch], cfg
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sch.step()
            ph = predict(model, cfg, val_passes)
            vm = stage_macro_mae(g[val_passes], ph)
            if vm < best[0]:
                best = (vm, epoch)
        del model, opt, sch
        torch.cuda.empty_cache()

        seed_all(seed)
        model = build_model(cfg, imagenet_state=weight).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        trainids = frame_ids(train_passes)
        for epoch in range(1, best[1] + 1):
            model.train()
            order = trainids[torch.randperm(len(trainids), device=device)]
            for batch in order.split(BATCH):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    o3, o5, op = model(batch_x(batch, True), mean, std)
                    loss = ordprog_loss(
                        o3, o5, op, c3[batch], c5[batch], targets[batch], cfg
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sch.step()
        ph = predict(model, cfg, test_passes)
        mae = stage_macro_mae(g[test_passes], ph)
        # save per-pass preds
        pred_path = OUT / f"proxy__{tag}__{variant_name}__seed{seed}_predictions.csv"
        pd.DataFrame(
            {
                "stage": g[test_passes].astype(int),
                "pass_id": keys[test_passes, 1].astype(int),
                "prediction": ph,
                "error": np.abs(ph - g[test_passes]),
            }
        ).to_csv(pred_path, index=False)
        out = dict(
            tag=tag,
            method=variant_name,
            seed=seed,
            selected_epochs=int(best[1]),
            inner_val_mae=float(best[0]),
            mae_stage_macro=float(mae),
            n_train_passes=int(len(train_passes)),
            n_test_passes=int(len(test_passes)),
            n_train_stages=int(len(np.unique(g[train_passes]))),
            n_test_stages=int(len(np.unique(g[test_passes]))),
        )
        result_path.write_text(json.dumps(out, indent=2))
        print("DONE", result_path.name, "MAE", round(mae, 3), flush=True)
        del model, opt, sch
        torch.cuda.empty_cache()
        return out

    results = []
    for tag, fold in folds.items():
        for method, variant in methods.items():
            for seed in SEEDS:
                results.append(
                    train_one(variant, seed, fold["train"], fold["test"], f"{tag}__{method}")
                )
    # summarize
    rows = []
    for r in results:
        if "mae_stage_macro" not in r:
            continue
        rows.append(
            dict(
                protocol=r["tag"].rsplit("__", 1)[0],
                method=r["tag"].rsplit("__", 1)[1],
                seed=r["seed"],
                mae=r["mae_stage_macro"],
                n_train_passes=r["n_train_passes"],
                n_test_passes=r["n_test_passes"],
            )
        )
    rdf = pd.DataFrame(rows)
    rdf.to_csv(TAB / "proxy_split_raw.csv", index=False)
    summ = (
        rdf.groupby(["protocol", "method"])["mae"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summ.to_csv(TAB / "proxy_split_summary.csv", index=False)
    print("PROXY_SPLITS\n", summ)
    return summ


# ---------- failure case board ----------
def run_failure_board(thr=5.0):
    audit = load_seedavg_preds("auditord__multi_gap_guard10__aord_erm__seed{seed}_predictions.csv")
    core = audit[audit.in_core].copy()
    core["abs_err"] = (core["ghat"] - core["stage"]).abs()
    core["alert"] = core["abs_err"] > thr
    # pick diverse stages
    alerts = core[core.alert].sort_values("abs_err", ascending=False)
    oks = core[~core.alert].sort_values("abs_err")
    picks = []
    # 2 large ALERT (re-check)
    for _, r in alerts.drop_duplicates("stage").head(2).iterrows():
        picks.append(dict(r, case="ALERT_large", action="Pause handover; re-check log vs frame/gate."))
    # 1 mid ALERT
    mid = alerts[(alerts.abs_err > thr) & (alerts.abs_err < 12)]
    if len(mid):
        r = mid.iloc[len(mid) // 2]
        picks.append(dict(r, case="ALERT_moderate", action="Short visual re-check; raise thr if chronic."))
    # 1 borderline OK
    near = oks.sort_values("abs_err", ascending=False).head(1)
    for _, r in near.iterrows():
        picks.append(dict(r, case="OK_borderline", action="Accept; monitor next pass."))
    # 1 clear OK
    clear = oks.sort_values("abs_err").head(1)
    for _, r in clear.iterrows():
        picks.append(dict(r, case="OK_clear", action="No action; gate supports log."))

    man = pd.read_csv(OUT / "sample_manifest.csv")
    # map pass -> first frame path via keys order in sampled_data
    data = np.load(OUT / "sampled_data.npz")
    stages, keys = data["stages"], data["keys"]
    key_to_pass = {(int(keys[i, 0]), int(keys[i, 1])): i for i in range(len(keys))}

    style()
    fig, axes = plt.subplots(2, len(picks), figsize=(1.55 * len(picks), 3.5))
    if len(picks) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    records = []
    for j, p in enumerate(picks):
        pi = key_to_pass[(int(p["stage"]), int(p["pass_id"]))]
        # 3 frames per pass; take middle
        frame_idx = pi * 3 + 1
        path = man.loc[frame_idx, "path"]
        rgb = Image.open(path).convert("RGB").resize((320, 96), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb).astype(np.float32) / 255.0
        gray = arr.mean(2)
        mask = (gray > 28 / 255).astype(np.float32)
        metal = arr * (0.25 + 0.75 * mask[..., None])
        metal_img = (np.clip(metal, 0, 1) * 255).astype(np.uint8)
        axes[0, j].imshow(rgb)
        axes[1, j].imshow(metal_img)
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(1.0)
                sp.set_color(BAD if p["case"].startswith("ALERT") else OK)
        flag = "ALERT" if p["case"].startswith("ALERT") else "OK"
        axes[0, j].set_title(
            f"{flag}\ng={int(p['stage'])} ĝ={p['ghat']:.1f}\n|Δ|={p['abs_err']:.1f}",
            fontsize=7.0,
            color=BAD if flag == "ALERT" else OK,
        )
        records.append(
            dict(
                case=p["case"],
                stage=int(p["stage"]),
                pass_id=int(p["pass_id"]),
                ghat=float(p["ghat"]),
                abs_err=float(p["abs_err"]),
                thr=thr,
                plant_action=p["action"],
                frame=str(path),
            )
        )
    axes[0, 0].set_ylabel("RGB", fontsize=7)
    axes[1, 0].set_ylabel("MetalField", fontsize=7)
    fig.suptitle(
        "Advisory cases on $k^*=10$ core (thr=5): gate supports human re-check",
        fontsize=8.2,
        color=INK,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG / "Fig30_alert_case_board.pdf", bbox_inches="tight")
    fig.savefig(FIG / "Fig30_alert_case_board.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(TAB / "alert_case_board.csv", index=False)
    print("FAILURE_BOARD", records)
    return records


def main():
    t0 = time.time()
    run_heuristics()
    run_self_consistency()
    run_camera_curve()
    summarize_leave_band()
    run_failure_board()
    train_proxy_folds()
    print("ALL_DONE", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
