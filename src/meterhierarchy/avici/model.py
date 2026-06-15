"""PyTorch AVICI model (Lorch et al., 2022, "Amortized Inference for Causal
Structure Learning").

The network alternates self-attention over the *variable* axis and the *sample*
axis (axial attention), pools over samples, and produces a directed edge
probability via a scaled inner product of two per-variable projections:

    theta_{i,j} = sigmoid( tau * <u_i, v_j> + b )

so ``theta_{i,j}`` is the probability of the directed edge ``i -> j`` (here:
``i`` is the parent of ``j``). The matrix is asymmetric by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AVICIConfig:
    dim: int = 128          # model width (k)
    n_layers: int = 8       # number of axial-attention blocks (L)
    n_heads: int = 8
    ffn_dim: int = 512
    dropout: float = 0.1
    in_channels: int = 2    # [standardized value, intervention indicator]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "AVICIConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AVICIBlock(nn.Module):
    """One axial layer: attend over variables (d), then over samples (n).

    Pre-norm transformer sublayers with residual connections.
    """

    def __init__(self, cfg: AVICIConfig):
        super().__init__()
        self.attn_d = nn.MultiheadAttention(cfg.dim, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.attn_n = nn.MultiheadAttention(cfg.dim, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ff_d = _FeedForward(cfg.dim, cfg.ffn_dim, cfg.dropout)
        self.ff_n = _FeedForward(cfg.dim, cfg.ffn_dim, cfg.dropout)
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.ln3 = nn.LayerNorm(cfg.dim)
        self.ln4 = nn.LayerNorm(cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (n, d, k)
        # --- attention across variables d (batch = samples n) ---
        h = self.ln1(x)
        a, _ = self.attn_d(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff_d(self.ln2(x)))

        # --- attention across samples n (batch = variables d) ---
        xt = x.transpose(0, 1)  # (d, n, k)
        h = self.ln3(xt)
        a, _ = self.attn_n(h, h, h, need_weights=False)
        xt = xt + self.drop(a)
        xt = xt + self.drop(self.ff_n(self.ln4(xt)))
        return xt.transpose(0, 1)  # (n, d, k)


class AVICIModel(nn.Module):
    """Maps an ``(n, d, 2)`` data tensor to ``(d, d)`` directed edge logits."""

    def __init__(self, cfg: AVICIConfig | None = None):
        super().__init__()
        cfg = cfg or AVICIConfig()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.in_channels, cfg.dim)
        self.blocks = nn.ModuleList([AVICIBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_out = nn.LayerNorm(cfg.dim)
        self.u_proj = nn.Linear(cfg.dim, cfg.dim)
        self.v_proj = nn.Linear(cfg.dim, cfg.dim)
        # Learned temperature (log-space) and bias for the edge head.
        self.log_scale = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.tensor(-3.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x ``(n, d, 2)``. Returns edge logits ``(d, d)`` (pre-sigmoid)."""
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        h = self.ln_out(h)
        z = h.max(dim=0).values  # (d, k): max-pool over samples -> permutation invariant in n
        u = F.normalize(self.u_proj(z), dim=-1)
        v = F.normalize(self.v_proj(z), dim=-1)
        logits = torch.exp(self.log_scale) * (u @ v.t()) + self.bias  # (d, d)
        return logits

    @torch.no_grad()
    def predict_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Edge probabilities ``(d, d)`` with a zeroed diagonal."""
        logits = self.forward(x)
        d = logits.shape[0]
        probs = torch.sigmoid(logits)
        probs = probs * (1.0 - torch.eye(d, device=probs.device, dtype=probs.dtype))
        return probs
