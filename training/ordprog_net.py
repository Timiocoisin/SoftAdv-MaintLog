"""OrdProg-Net: ResNet-18 + CORAL + progress + appearance prior.

v1: 4th-channel metal mask (hurts ImageNet conv1 when combined with CORAL).
v2: keep 3-ch ResNet; late-fuse unsupervised metal ratios into heads (preferred).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


METAL_THRESHOLD_8BIT = 28.0


@dataclass(frozen=True)
class OrdProgConfig:
    name: str
    use_appearance: bool = True
    use_coral: bool = True
    use_corn: bool = False  # CORN conditional ordinal (Shi et al.); mutually exclusive w/ CORAL CE5
    # "off" | "input4" | "late" | "mask1" | "metal_op" | "free_op" | "audit_gate" | "audit_ord"
    # metal_op: constrained MetalField operator modulating RGB (keeps 3-ch ResNet)
    # free_op: unconstrained spatial gate (negative control; not luminance-τ)
    # audit_gate: FloorRefine under fixed luminance veto m0 (operator upgrade)
    # audit_ord: dual-pathway Audit-OrdNet (RGB stem + audited MetalField stem + cross-FiLM)
    appearance_mode: str = "late"
    ce3_weight: float = 1.0
    ce5_weight: float = 1.0
    coral_weight: float = 1.0
    progress_weight: float = 2.0
    progress_loss: str = "mse"  # "mse" | "smooth_l1"
    metal_threshold: float = METAL_THRESHOLD_8BIT / 255.0
    # Bright-metal gate: fixed τ | per-image Otsu | learnable scalar τ (still luminance-thresholded)
    metal_gate_mode: str = "fixed"  # "fixed" | "otsu" | "learnable"
    soft_mask_temperature: float = 0.02
    # Residual floor for metal_op / free_op: x' = x ⊙ (λ + (1-λ) m)
    metal_floor: float = 0.25
    metal_emb: int = 64
    head_hidden: int = 256
    tta_flip: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def _variant(
    name: str,
    *,
    use_coral: bool,
    appearance_mode: str,
    progress_weight: float = 2.0,
    coral_weight: float = 1.0,
    ce3_weight: float = 1.0,
    ce5_weight: float = 1.0,
    progress_loss: str = "mse",
    tta_flip: bool = False,
    metal_threshold: float = METAL_THRESHOLD_8BIT / 255.0,
    metal_gate_mode: str = "fixed",
    soft_mask_temperature: float = 0.02,
    metal_floor: float = 0.25,
    use_corn: bool = False,
) -> OrdProgConfig:
    return OrdProgConfig(
        name=name,
        use_appearance=appearance_mode != "off",
        use_coral=use_coral,
        use_corn=use_corn,
        appearance_mode=appearance_mode,
        progress_weight=progress_weight,
        coral_weight=coral_weight,
        ce3_weight=ce3_weight,
        ce5_weight=ce5_weight,
        progress_loss=progress_loss,
        tta_flip=tta_flip,
        metal_threshold=metal_threshold,
        metal_gate_mode=metal_gate_mode,
        soft_mask_temperature=soft_mask_temperature,
        metal_floor=metal_floor,
    )


# v1 names kept for compatibility with finished runs
VARIANTS: Dict[str, OrdProgConfig] = {
    "baseline_ce": _variant("baseline_ce", use_coral=False, appearance_mode="off"),
    "coral_prog": _variant("coral_prog", use_coral=True, appearance_mode="off"),
    "appearance_ce": _variant("appearance_ce", use_coral=False, appearance_mode="input4"),
    "ordprog_full": _variant("ordprog_full", use_coral=True, appearance_mode="input4"),
    # dense progress regression (RGB only; ProgressNet-style continuous head)
    "prog_only": _variant(
        "prog_only",
        use_coral=False,
        appearance_mode="off",
        ce3_weight=0.0,
        ce5_weight=0.0,
        progress_weight=2.0,
    ),
    # ProgressNet-style RGB continuous progress (matched LG budget; smooth-L1)
    "progressnet_rgb": _variant(
        "progressnet_rgb",
        use_coral=False,
        appearance_mode="off",
        ce3_weight=0.0,
        ce5_weight=0.0,
        progress_weight=3.0,
        progress_loss="smooth_l1",
    ),
    # CORN ordinal (RGB; 5 stage bins; conditional rank probs) + light progress
    "corn_rgb": _variant(
        "corn_rgb",
        use_coral=False,
        use_corn=True,
        appearance_mode="off",
        ce3_weight=0.0,
        ce5_weight=0.0,
        coral_weight=1.0,
        progress_weight=1.0,
        progress_loss="smooth_l1",
    ),
    # metal-mask-only CNN controls (no RGB; same heads as full / appearance_ce)
    "mask_only_coral": _variant("mask_only_coral", use_coral=True, appearance_mode="mask1"),
    "mask_only_ce": _variant("mask_only_ce", use_coral=False, appearance_mode="mask1"),
    # novelty-boost head controls (metal channel; same ResNet-18 stem)
    "prog_only_metal": _variant(
        "prog_only_metal",
        use_coral=False,
        appearance_mode="input4",
        ce3_weight=0.0,
        ce5_weight=0.0,
        progress_weight=2.0,
    ),
    "ce5_prog_metal": _variant(
        "ce5_prog_metal",
        use_coral=False,
        appearance_mode="input4",
        ce3_weight=0.0,
        ce5_weight=1.0,
        progress_weight=2.0,
    ),
    # Auditable bright-metal gates (still luminance thresholds; not free attention)
    "ordprog_otsu": _variant(
        "ordprog_otsu", use_coral=True, appearance_mode="input4", metal_gate_mode="otsu"
    ),
    "appearance_otsu": _variant(
        "appearance_otsu", use_coral=False, appearance_mode="input4", metal_gate_mode="otsu"
    ),
    "ordprog_learn_tau": _variant(
        "ordprog_learn_tau",
        use_coral=True,
        appearance_mode="input4",
        metal_gate_mode="learnable",
    ),
    "appearance_learn_tau": _variant(
        "appearance_learn_tau",
        use_coral=False,
        appearance_mode="input4",
        metal_gate_mode="learnable",
    ),
    # B-upgrade: MetalField structural operator (3-ch; not 4th-channel concat)
    "ordprog_metal_op": _variant(
        "ordprog_metal_op", use_coral=True, appearance_mode="metal_op", metal_gate_mode="fixed"
    ),
    "appearance_metal_op": _variant(
        "appearance_metal_op", use_coral=False, appearance_mode="metal_op", metal_gate_mode="fixed"
    ),
    "ordprog_metal_op_learn": _variant(
        "ordprog_metal_op_learn",
        use_coral=True,
        appearance_mode="metal_op",
        metal_gate_mode="learnable",
    ),
    "appearance_metal_op_learn": _variant(
        "appearance_metal_op_learn",
        use_coral=False,
        appearance_mode="metal_op",
        metal_gate_mode="learnable",
    ),
    # Negative control: unconstrained spatial gate (same modulation algebra)
    "ordprog_free_op": _variant(
        "ordprog_free_op", use_coral=True, appearance_mode="free_op", metal_gate_mode="fixed"
    ),
    "appearance_free_op": _variant(
        "appearance_free_op", use_coral=False, appearance_mode="free_op", metal_gate_mode="fixed"
    ),
    # Route-1: AuditGate — learnable spatial refine under luminance veto m0
    "ordprog_audit_gate": _variant(
        "ordprog_audit_gate",
        use_coral=True,
        appearance_mode="audit_gate",
        metal_gate_mode="learnable",
    ),
    "appearance_audit_gate": _variant(
        "appearance_audit_gate",
        use_coral=False,
        appearance_mode="audit_gate",
        metal_gate_mode="learnable",
    ),
    # Route-A: Audit-OrdNet dual pathway (architecture experiment; not yet in manuscript)
    "audit_ord_coral": _variant(
        "audit_ord_coral",
        use_coral=True,
        appearance_mode="audit_ord",
        metal_gate_mode="learnable",
    ),
    "audit_ord_ce": _variant(
        "audit_ord_ce",
        use_coral=False,
        appearance_mode="audit_ord",
        metal_gate_mode="learnable",
    ),
    # v2 search
    "v2_late_coral": _variant(
        "v2_late_coral", use_coral=True, appearance_mode="late", progress_weight=2.0
    ),
    "v2_late_prog3": _variant(
        "v2_late_prog3",
        use_coral=True,
        appearance_mode="late",
        progress_weight=3.0,
        coral_weight=1.25,
        ce3_weight=0.5,
        progress_loss="smooth_l1",
    ),
    "v2_late_prog4_tta": _variant(
        "v2_late_prog4_tta",
        use_coral=True,
        appearance_mode="late",
        progress_weight=4.0,
        coral_weight=1.5,
        ce3_weight=0.5,
        progress_loss="smooth_l1",
        tta_flip=True,
    ),
    "v2_late_ce": _variant(
        "v2_late_ce", use_coral=False, appearance_mode="late", progress_weight=3.0, progress_loss="smooth_l1"
    ),
}


def rgb_to_gray(x_01: torch.Tensor) -> torch.Tensor:
    r, g, b = x_01[:, 0:1], x_01[:, 1:2], x_01[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def metal_mask(x_01: torch.Tensor, threshold: float) -> torch.Tensor:
    return (rgb_to_gray(x_01) > threshold).to(dtype=x_01.dtype)


def otsu_thresholds(gray_01: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
    """Per-image Otsu threshold on luminance in [0,1]. Returns shape (N,1,1,1)."""
    n = gray_01.shape[0]
    flat = gray_01.reshape(n, -1).clamp(0, 1)
    # Histogram counts
    idx = (flat * (n_bins - 1)).long()
    hist = torch.zeros(n, n_bins, device=gray_01.device, dtype=torch.float32)
    hist.scatter_add_(1, idx, torch.ones_like(flat, dtype=torch.float32))
    p = hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)
    omega = torch.cumsum(p, dim=1)
    mu = torch.cumsum(p * torch.arange(n_bins, device=gray_01.device).float(), dim=1)
    mu_t = mu[:, -1:].clamp_min(1e-8)
    sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega).clamp_min(1e-8))
    sigma_b[:, 0] = 0
    sigma_b[:, -1] = 0
    t = sigma_b.argmax(dim=1).float() / float(n_bins - 1)
    return t.view(n, 1, 1, 1)


def bright_metal_mask(
    x_01: torch.Tensor,
    *,
    mode: str,
    threshold: float,
    tau_param: Optional[torch.Tensor] = None,
    temperature: float = 0.02,
) -> torch.Tensor:
    """Auditable bright-metal gate: hard/soft threshold on luminance only."""
    gray = rgb_to_gray(x_01)
    if mode == "otsu":
        thr = otsu_thresholds(gray)
        return (gray > thr).to(dtype=x_01.dtype)
    if mode == "learnable":
        assert tau_param is not None
        tau = torch.sigmoid(tau_param).view(1, 1, 1, 1)
        # Soft mask keeps τ differentiable; still a monotone luminance gate.
        return torch.sigmoid((gray - tau) / max(float(temperature), 1e-4))
    return (gray > threshold).to(dtype=x_01.dtype)


def metal_ratio_feats(x_01: torch.Tensor, threshold: float) -> torch.Tensor:
    """Unsupervised 4-D metal ratios: full / left / centre / right thirds."""
    gray = rgb_to_gray(x_01)
    mask = (gray > threshold).to(dtype=x_01.dtype)
    # mask: N,1,H,W
    n, _, h, w = mask.shape
    w3 = w // 3
    regions = [
        mask,
        mask[:, :, :, :w3],
        mask[:, :, :, w3 : 2 * w3],
        mask[:, :, :, 2 * w3 :],
    ]
    feats = [r.mean(dim=(1, 2, 3)) for r in regions]
    return torch.stack(feats, dim=1)  # N,4


class OrdProgNet(nn.Module):
    def __init__(self, cfg: OrdProgConfig, imagenet_state: Optional[Dict] = None):
        super().__init__()
        self.cfg = cfg
        backbone = models.resnet18(weights=None)
        if imagenet_state is not None:
            backbone.load_state_dict(imagenet_state)

        if cfg.appearance_mode == "input4":
            old = backbone.conv1
            new = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new.weight[:, :3].copy_(old.weight)
                new.weight[:, 3:4].copy_(old.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = new
        elif cfg.appearance_mode == "mask1":
            # Single-channel bright-metal mask; init from ImageNet RGB filter mean.
            old = backbone.conv1
            new = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = new
        # metal_op / free_op keep the ImageNet 3-ch conv1 intact.

        backbone.fc = nn.Identity()
        self.backbone = backbone

        feat_dim = 512
        if cfg.appearance_mode == "late":
            self.metal_mlp = nn.Sequential(
                nn.Linear(4, cfg.metal_emb),
                nn.ReLU(inplace=True),
                nn.Linear(cfg.metal_emb, cfg.metal_emb),
                nn.ReLU(inplace=True),
            )
            feat_dim = 512 + cfg.metal_emb
            # FiLM-lite: scale RGB embedding from metal
            self.metal_gate = nn.Sequential(nn.Linear(cfg.metal_emb, 512), nn.Sigmoid())
        else:
            self.metal_mlp = None
            self.metal_gate = None

        # Unconstrained spatial gate control (B negative control): 1x1 on luminance.
        if cfg.appearance_mode == "free_op":
            self.free_gate = nn.Conv2d(1, 1, kernel_size=1, bias=True)
            nn.init.zeros_(self.free_gate.weight)
            nn.init.zeros_(self.free_gate.bias)  # sigmoid(0)=0.5 at init
        else:
            self.free_gate = None

        # AuditGate FloorRefine (v3): Φ-locked gate m:=m0; learn spatial MetalField floor.
        # λ_map = clamp(λ0 + β·tanh(refine(I_gray)), λ_lo, λ_hi); x' = x ⊙ (λ + (1-λ) m0).
        # At init refine≈0 → λ≈λ0, identical to metal_op; β grows only if useful.
        if cfg.appearance_mode == "audit_gate":
            self.audit_refine = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(1, 1, kernel_size=1, bias=True),
            )
            nn.init.kaiming_normal_(self.audit_refine[0].weight, nonlinearity="relu")
            nn.init.zeros_(self.audit_refine[0].bias)
            nn.init.kaiming_normal_(self.audit_refine[2].weight, nonlinearity="relu")
            nn.init.zeros_(self.audit_refine[2].bias)
            nn.init.zeros_(self.audit_refine[-1].weight)
            nn.init.zeros_(self.audit_refine[-1].bias)  # tanh(0)=0 → λ=λ0 at init
            # amplitude β of floor residual; softplus(β_raw) init ≈ 0.12
            self.audit_beta_raw = nn.Parameter(torch.tensor(-2.0))  # softplus(-2)≈0.127
            self.register_parameter("audit_gamma_logit", None)  # retired (v2 residual mix)
        else:
            self.audit_refine = None
            self.register_parameter("audit_beta_raw", None)
            self.register_parameter("audit_gamma_logit", None)

        def head(out_dim: int) -> nn.Module:
            if cfg.head_hidden and feat_dim != 512:
                return nn.Sequential(
                    nn.Linear(feat_dim, cfg.head_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(cfg.head_hidden, out_dim),
                )
            return nn.Linear(feat_dim, out_dim)

        self.head3 = head(3)
        self.head5 = head(4 if (cfg.use_coral or cfg.use_corn) else 5)
        self.progress = head(1)

        # Learnable bright-metal τ in (0,1); init near fixed laboratory heuristic 28/255.
        if cfg.metal_gate_mode == "learnable" and cfg.appearance_mode != "free_op":
            init = float(cfg.metal_threshold)
            init = min(max(init, 1e-3), 1.0 - 1e-3)
            logit = torch.log(torch.tensor(init / (1.0 - init)))
            self.tau_logit = nn.Parameter(logit)
        else:
            self.register_parameter("tau_logit", None)

    def _metal(self, x_01: torch.Tensor) -> torch.Tensor:
        return bright_metal_mask(
            x_01,
            mode=self.cfg.metal_gate_mode,
            threshold=self.cfg.metal_threshold,
            tau_param=self.tau_logit,
            temperature=self.cfg.soft_mask_temperature,
        )

    def learned_tau(self) -> Optional[float]:
        if self.tau_logit is None:
            return None
        return float(torch.sigmoid(self.tau_logit.detach()).cpu())

    def _apply_metal_field(self, x_01: torch.Tensor, metal: torch.Tensor) -> torch.Tensor:
        """Structural MetalField operator: x' = x ⊙ (λ + (1-λ) m). Keeps 3 RGB channels."""
        lam = float(self.cfg.metal_floor)
        lam = min(max(lam, 0.0), 1.0)
        return x_01 * (lam + (1.0 - lam) * metal)

    def audit_gamma(self) -> Optional[float]:
        """Backward-compatible alias: report floor-amplitude β (not mix-γ)."""
        if getattr(self, "audit_beta_raw", None) is None:
            return None
        return float(F.softplus(self.audit_beta_raw.detach()).cpu())

    def audit_lambda_map(self, x_01: torch.Tensor) -> torch.Tensor:
        """Spatially varying MetalField floor under AuditGate FloorRefine."""
        assert self.audit_refine is not None and self.audit_beta_raw is not None
        gray = rgb_to_gray(x_01)
        delta = torch.tanh(self.audit_refine(gray))
        beta = F.softplus(self.audit_beta_raw)
        lam0 = float(self.cfg.metal_floor)
        return (lam0 + beta * delta).clamp(0.05, 0.60)

    def audit_gate_map(self, x_01: torch.Tensor) -> torch.Tensor:
        """AuditGate FloorRefine: auditable gate is luminance-τ veto m0 (Φ-locked)."""
        return self._metal(x_01)

    def gate_map(self, x_01: torch.Tensor) -> torch.Tensor:
        """Expose the active appearance gate for Φ audits."""
        mode = self.cfg.appearance_mode
        if mode == "audit_gate":
            return self.audit_gate_map(x_01)
        if mode == "free_op":
            return torch.sigmoid(self.free_gate(rgb_to_gray(x_01)))
        if mode in ("metal_op", "input4", "mask1"):
            return self._metal(x_01)
        gray = rgb_to_gray(x_01)
        return (gray > self.cfg.metal_threshold).to(dtype=x_01.dtype)

    def _backbone_rgb(self, x_01: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.cfg.appearance_mode == "input4":
            metal = self._metal(x_01)
            x = torch.cat([(x_01 - mean) / std, metal], dim=1)
        elif self.cfg.appearance_mode == "mask1":
            # Binary/soft mask in [0,1]; centre for stable transfer from ImageNet-mean init.
            metal = self._metal(x_01)
            x = (metal - 0.5) / 0.5
        elif self.cfg.appearance_mode == "metal_op":
            metal = self._metal(x_01)
            x_mod = self._apply_metal_field(x_01, metal)
            x = (x_mod - mean) / std
        elif self.cfg.appearance_mode == "audit_gate":
            metal = self.audit_gate_map(x_01)  # m0; Φ-locked
            lam = self.audit_lambda_map(x_01)
            x_mod = x_01 * (lam + (1.0 - lam) * metal)
            x = (x_mod - mean) / std
        elif self.cfg.appearance_mode == "free_op":
            gray = rgb_to_gray(x_01)
            metal = torch.sigmoid(self.free_gate(gray))
            x_mod = self._apply_metal_field(x_01, metal)
            x = (x_mod - mean) / std
        else:
            x = (x_01 - mean) / std
        return self.backbone(x)

    def encode(self, x_01: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        z = self._backbone_rgb(x_01, mean, std)
        if self.cfg.appearance_mode == "late":
            # Late fusion keeps the fixed heuristic ratios for compatibility.
            m = metal_ratio_feats(x_01, self.cfg.metal_threshold)
            me = self.metal_mlp(m)
            z = z * (1.0 + self.metal_gate(me))
            z = torch.cat([z, me], dim=1)
        return z

    def forward(
        self, x_01: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x_01, mean, std)
        o3 = self.head3(z)
        o5 = self.head5(z)
        op = torch.sigmoid(self.progress(z)).squeeze(1)
        return o3, o5, op


def _build_resnet18(imagenet_state: Optional[Dict] = None) -> nn.Module:
    backbone = models.resnet18(weights=None)
    if imagenet_state is not None:
        backbone.load_state_dict(imagenet_state)
    backbone.fc = nn.Identity()
    return backbone


class AuditOrdNet(nn.Module):
    """Audit-OrdNet (Route A): dual-pathway audited ordinal architecture.

    Pathway R: ImageNet ResNet-18 on RGB.
    Pathway A: ImageNet ResNet-18 on MetalField FloorRefine view
               (gate m:=m0 Φ-locked; spatial floor λ=λ0+β·tanh(refine)).
    Cross-FiLM: each embedding modulates the other, then concat → task heads.

    This is a structural network (two stems + constrained cross-talk), not a
    single-stem recipe with an input prior. Interface matches OrdProgNet.
    """

    def __init__(self, cfg: OrdProgConfig, imagenet_state: Optional[Dict] = None):
        super().__init__()
        if cfg.appearance_mode != "audit_ord":
            raise ValueError(f"AuditOrdNet requires appearance_mode='audit_ord', got {cfg.appearance_mode}")
        self.cfg = cfg

        self.backbone_rgb = _build_resnet18(imagenet_state)
        # Second stem: same ImageNet init (independent weights after copy)
        self.backbone_audit = _build_resnet18(imagenet_state)

        # FloorRefine on luminance (shared with AuditGate-F semantics)
        self.audit_refine = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=1, bias=True),
        )
        nn.init.kaiming_normal_(self.audit_refine[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.audit_refine[0].bias)
        nn.init.kaiming_normal_(self.audit_refine[2].weight, nonlinearity="relu")
        nn.init.zeros_(self.audit_refine[2].bias)
        nn.init.zeros_(self.audit_refine[-1].weight)
        nn.init.zeros_(self.audit_refine[-1].bias)
        self.audit_beta_raw = nn.Parameter(torch.tensor(0.0))  # softplus(0)≈0.69 wide default

        # Cross-FiLM: near-identity at init (last layer zeros)
        self.cross_r = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, 512))
        self.cross_a = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, 512))
        nn.init.zeros_(self.cross_r[-1].weight)
        nn.init.zeros_(self.cross_r[-1].bias)
        nn.init.zeros_(self.cross_a[-1].weight)
        nn.init.zeros_(self.cross_a[-1].bias)

        feat_dim = 1024  # concat of two 512-D embeddings
        hidden = cfg.head_hidden or 256

        def head(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(feat_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, out_dim),
            )

        self.head3 = head(3)
        self.head5 = head(4 if (cfg.use_coral or cfg.use_corn) else 5)
        self.progress = head(1)

        init = float(cfg.metal_threshold)
        init = min(max(init, 1e-3), 1.0 - 1e-3)
        logit = torch.log(torch.tensor(init / (1.0 - init)))
        self.tau_logit = nn.Parameter(logit) if cfg.metal_gate_mode == "learnable" else None
        if self.tau_logit is None:
            self.register_parameter("tau_logit", None)

    def _metal(self, x_01: torch.Tensor) -> torch.Tensor:
        return bright_metal_mask(
            x_01,
            mode=self.cfg.metal_gate_mode if self.tau_logit is not None else "fixed",
            threshold=self.cfg.metal_threshold,
            tau_param=self.tau_logit,
            temperature=self.cfg.soft_mask_temperature,
        )

    def learned_tau(self) -> Optional[float]:
        if self.tau_logit is None:
            return None
        return float(torch.sigmoid(self.tau_logit.detach()).cpu())

    def audit_gamma(self) -> Optional[float]:
        return float(F.softplus(self.audit_beta_raw.detach()).cpu())

    def audit_lambda_map(self, x_01: torch.Tensor) -> torch.Tensor:
        gray = rgb_to_gray(x_01)
        delta = torch.tanh(self.audit_refine(gray))
        beta = F.softplus(self.audit_beta_raw)
        lam0 = float(self.cfg.metal_floor)
        return (lam0 + beta * delta).clamp(0.05, 0.60)

    def gate_map(self, x_01: torch.Tensor) -> torch.Tensor:
        """Φ-locked auditable gate = luminance-τ veto m0."""
        return self._metal(x_01)

    def encode(self, x_01: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        m0 = self._metal(x_01)
        lam = self.audit_lambda_map(x_01)
        x_audit = x_01 * (lam + (1.0 - lam) * m0)

        z_r = self.backbone_rgb((x_01 - mean) / std)
        z_a = self.backbone_audit((x_audit - mean) / std)

        # Bidirectional Cross-FiLM (near-identity at init)
        scale_r = torch.sigmoid(self.cross_r(z_a))
        scale_a = torch.sigmoid(self.cross_a(z_r))
        z_r = z_r * (1.0 + scale_r)
        z_a = z_a * (1.0 + scale_a)
        return torch.cat([z_r, z_a], dim=1)

    def forward(
        self, x_01: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x_01, mean, std)
        o3 = self.head3(z)
        o5 = self.head5(z)
        op = torch.sigmoid(self.progress(z)).squeeze(1)
        return o3, o5, op


def build_model(cfg: OrdProgConfig, imagenet_state: Optional[Dict] = None) -> nn.Module:
    """Factory: Audit-OrdNet for audit_ord mode, else OrdProgNet."""
    if cfg.appearance_mode == "audit_ord":
        return AuditOrdNet(cfg, imagenet_state=imagenet_state)
    return OrdProgNet(cfg, imagenet_state=imagenet_state)


def coral_bce_loss(logits: torch.Tensor, levels: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
    targets = torch.stack([(levels > k).float() for k in range(n_classes - 1)], dim=1)
    return F.binary_cross_entropy_with_logits(logits, targets)


def coral_levels_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits).sum(dim=1)


def corn_bce_loss(logits: torch.Tensor, levels: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
    """CORN: conditional P(y>k | y>k-1) with importance weighting by rank mass."""
    # Conditional targets: for sample with label y, train z_k only when y > k-1
    # Equivalent importance-weighted form used in CORN reference impls.
    n_thr = n_classes - 1
    losses = []
    for k in range(n_thr):
        # samples that reach threshold k (y > k-1); for k=0 all samples
        if k == 0:
            mask = torch.ones_like(levels, dtype=torch.bool)
        else:
            mask = levels > (k - 1)
        if mask.any():
            tgt = (levels[mask] > k).float()
            losses.append(F.binary_cross_entropy_with_logits(logits[mask, k], tgt))
    if not losses:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return torch.stack(losses).mean()


def corn_levels_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Decode CORN: P(y>k)=∏_{j≤k} σ(z_j); rank = Σ 1[P(y>k)>0.5]."""
    cond = torch.sigmoid(logits)
    # cumprod along thresholds
    cum = torch.cumprod(cond, dim=1)
    return (cum > 0.5).float().sum(dim=1)


def corn_progress_from_logits(logits: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
    """Soft expected rank in [0,1] from CORN cumulative probs (for MAE ranking)."""
    cond = torch.sigmoid(logits)
    cum = torch.cumprod(cond, dim=1)  # P(y>k)
    # P(y=0)=1-P(y>0); P(y=k)=P(y>k-1)-P(y>k); P(y=K-1)=P(y>K-2)
    p0 = 1.0 - cum[:, 0]
    parts = [p0.unsqueeze(1)]
    for k in range(1, n_classes - 1):
        parts.append((cum[:, k - 1] - cum[:, k]).unsqueeze(1))
    parts.append(cum[:, -1].unsqueeze(1))
    probs = torch.cat(parts, dim=1).clamp_min(0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    ranks = torch.arange(n_classes, device=logits.device, dtype=logits.dtype)
    exp = (probs * ranks).sum(dim=1)
    return exp / float(n_classes - 1)


def ordprog_loss(
    o3: torch.Tensor,
    o5: torch.Tensor,
    op: torch.Tensor,
    y3: torch.Tensor,
    y5: torch.Tensor,
    yp: torch.Tensor,
    cfg: OrdProgConfig,
) -> torch.Tensor:
    loss = torch.zeros((), device=op.device, dtype=op.dtype)
    if cfg.ce3_weight:
        loss = loss + cfg.ce3_weight * F.cross_entropy(o3, y3)
    if cfg.progress_weight:
        if cfg.progress_loss == "smooth_l1":
            loss = loss + cfg.progress_weight * F.smooth_l1_loss(op, yp)
        else:
            loss = loss + cfg.progress_weight * F.mse_loss(op, yp)
    if cfg.use_corn:
        if cfg.coral_weight:
            loss = loss + cfg.coral_weight * corn_bce_loss(o5, y5, n_classes=5)
    elif cfg.use_coral:
        if cfg.coral_weight:
            loss = loss + cfg.coral_weight * coral_bce_loss(o5, y5, n_classes=5)
    elif cfg.ce5_weight:
        loss = loss + cfg.ce5_weight * F.cross_entropy(o5, y5)
    return loss


@torch.no_grad()
def decode_predictions(
    o3: torch.Tensor, o5: torch.Tensor, op: torch.Tensor, cfg: OrdProgConfig
) -> Dict[str, torch.Tensor]:
    p3 = o3.argmax(dim=1)
    if cfg.use_corn:
        p5 = corn_levels_from_logits(o5).round().clamp(0, 4).long()
        # Continuous MAE: use progress head when trained; else CORN soft expectation
        progress = op if cfg.progress_weight > 0 else corn_progress_from_logits(o5, n_classes=5)
    elif cfg.use_coral:
        p5 = coral_levels_from_logits(o5).round().clamp(0, 4).long()
        progress = op
    else:
        p5 = o5.argmax(dim=1)
        progress = op
    return {"y3": p3, "y5": p5, "progress": progress}


V2_VARIANTS = [k for k in VARIANTS if k.startswith("v2_")]
