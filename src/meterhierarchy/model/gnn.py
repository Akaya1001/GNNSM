"""Two-pass directed-attention GNN for meter hierarchy detection.

Architecture highlights:

1. **Dual embeddings (DEDGAT):** every node carries separate parent/child
   projections, so attention is *directed* (asymmetric) unlike standard GAT.
2. **Jumping knowledge:** all layer outputs feed the classifier (anti
   over-smoothing).
3. **NOTEARS acyclicity** term as a differentiable regularizer.
4. **Enhanced edge updater** with sibling/parent context.
5. **Two-pass wrapper:** a small first pass produces 6 latent edge features that
   condition a deeper second pass.

Every node also receives Laplacian positional encodings (computed in
:mod:`meterhierarchy.features`).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Directed dual-embedding attention layer (DEDGAT)
# ---------------------------------------------------------------------------
class DualEmbeddingAttention(nn.Module):
    """Directed message passing with separate parent/child embeddings."""

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.W_parent_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_parent_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_parent_v = nn.Linear(hidden_dim, hidden_dim)

        self.W_child_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_child_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_child_v = nn.Linear(hidden_dim, hidden_dim)

        self.W_edge_attn = nn.Linear(hidden_dim, n_heads)

        self.gate_parent = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.gate_child = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_emb: torch.Tensor, edge_emb: torch.Tensor) -> torch.Tensor:
        N, D = node_emb.shape
        H = self.n_heads
        d_h = self.head_dim

        mask = torch.eye(N, device=node_emb.device, dtype=torch.bool)

        # Outgoing stream: node i as parent attends over all j as children.
        Q_out = self.W_parent_q(node_emb).view(N, H, d_h)
        K_out = self.W_child_k(node_emb).view(N, H, d_h)
        V_out = self.W_child_v(node_emb).view(N, H, d_h)
        attn_out = torch.einsum("ihd,jhd->ijh", Q_out, K_out) / math.sqrt(d_h)
        attn_out = attn_out + self.W_edge_attn(edge_emb)
        attn_out = attn_out.masked_fill(mask.unsqueeze(-1), -6e4)
        attn_out = F.softmax(attn_out, dim=1)
        attn_out = self.dropout(attn_out)
        msg_out = torch.einsum("ijh,jhd->ihd", attn_out, V_out).reshape(N, D)

        # Incoming stream: node i as child attends over all j as parents.
        Q_in = self.W_child_q(node_emb).view(N, H, d_h)
        K_in = self.W_parent_k(node_emb).view(N, H, d_h)
        V_in = self.W_parent_v(node_emb).view(N, H, d_h)
        attn_in = torch.einsum("ihd,jhd->ijh", Q_in, K_in) / math.sqrt(d_h)
        attn_in = attn_in + self.W_edge_attn(edge_emb).transpose(0, 1)
        attn_in = attn_in.masked_fill(mask.unsqueeze(-1), -6e4)
        attn_in = F.softmax(attn_in, dim=1)
        attn_in = self.dropout(attn_in)
        msg_in = torch.einsum("ijh,jhd->ihd", attn_in, V_in).reshape(N, D)

        # Gated fusion of the two directional streams + transformer-style block.
        g_out = self.gate_parent(torch.cat([node_emb, msg_out], dim=-1))
        g_in = self.gate_child(torch.cat([node_emb, msg_in], dim=-1))
        fused = self.out_proj(torch.cat([g_out * msg_out, g_in * msg_in], dim=-1))

        node_emb = self.norm1(node_emb + self.dropout(fused))
        node_emb = self.norm2(node_emb + self.dropout(self.ffn(node_emb)))
        return node_emb


# ---------------------------------------------------------------------------
# Edge updater with sibling/parent context
# ---------------------------------------------------------------------------
class EnhancedEdgeUpdater(nn.Module):
    """Refine each edge embedding using its endpoints and neighborhood context."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, edge_emb: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        N, D = node_emb.shape
        src = node_emb.unsqueeze(1).expand(N, N, D)
        dst = node_emb.unsqueeze(0).expand(N, N, D)
        sibling_ctx = edge_emb.mean(dim=1, keepdim=True).expand(N, N, D)
        parent_ctx = edge_emb.mean(dim=0, keepdim=True).expand(N, N, D)
        combined = torch.cat([edge_emb, src, dst, sibling_ctx, parent_ctx], dim=-1)
        update = self.net(combined)
        return self.norm(edge_emb + update)


# ---------------------------------------------------------------------------
# NOTEARS acyclicity constraint (Zheng et al., 2018)
# ---------------------------------------------------------------------------
def acyclicity_constraint(probs: torch.Tensor) -> torch.Tensor:
    """``h(W) = tr(exp(W o W)) - N`` with ``W = probs``.

    The Hadamard square happens inside the exponential series. A 4th-order Taylor
    approximation of the matrix exponential is sufficient for ``N <= 300`` with
    ``probs`` in ``[0, 1]``.
    """
    N = probs.shape[0]
    W = probs
    M = W * W  # element-wise (Hadamard) square
    M2 = M @ M
    M3 = M2 @ M
    M4 = M3 @ M
    exp_approx = torch.eye(N, device=probs.device) + M + M2 / 2 + M3 / 6 + M4 / 24
    return torch.trace(exp_approx) - N


# ---------------------------------------------------------------------------
# Latent edge features bridging pass 1 and pass 2
# ---------------------------------------------------------------------------
def compute_latent_edge_features(edge_logits_p1: torch.Tensor) -> torch.Tensor:
    """Six differentiable edge features derived from pass-1 probabilities.

    Returns an ``(N, N, 6)`` tensor:
        0: raw probability ``p[i, j]``
        1: asymmetry ``p[i, j] - p[j, i]``
        2: cycle indicator ``p[i, j] * p[j, i]``
        3: child competition ``p[i, j] / sum_k p[i, k]``
        4: parent competition ``p[i, j] / sum_k p[k, j]``
        5: row z-score of the parent over all children
    """
    N = edge_logits_p1.shape[0]
    p = torch.sigmoid(edge_logits_p1)
    mask = 1.0 - torch.eye(N, device=p.device, dtype=p.dtype)
    p = p * mask

    row_sum = p.sum(dim=1, keepdim=True) + 1e-6
    col_sum = p.sum(dim=0, keepdim=True) + 1e-6
    row_mean = p.mean(dim=1, keepdim=True)
    row_std = p.std(dim=1, keepdim=True) + 1e-6

    f0 = p
    f1 = p - p.T
    f2 = p * p.T
    f3 = p / row_sum
    f4 = p / col_sum
    f5 = (p - row_mean) / row_std

    return torch.stack([f0, f1, f2, f3, f4, f5], dim=-1)  # (N, N, 6)


# ---------------------------------------------------------------------------
# Single-pass GNN block
# ---------------------------------------------------------------------------
class MeterHierarchyGNN(nn.Module):
    """One GNN block: encoder -> L (attention + edge-update) layers -> 3 heads."""

    def __init__(
        self,
        n_edge_features: int = 58,
        n_node_features: int = 15,
        n_pe_features: int = 8,
        hidden_dim: int = 128,
        n_layers: int = 5,
        n_heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(n_node_features + n_pe_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(n_edge_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.mp_layers = nn.ModuleList(
            [DualEmbeddingAttention(hidden_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.edge_updaters = nn.ModuleList(
            [EnhancedEdgeUpdater(hidden_dim, dropout) for _ in range(n_layers)]
        )

        # Jumping knowledge: combine encoder + every layer output.
        self.jk_node_proj = nn.Linear(hidden_dim * (n_layers + 1), hidden_dim)
        self.jk_edge_proj = nn.Linear(hidden_dim * (n_layers + 1), hidden_dim)

        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.root_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, node_feats: torch.Tensor, edge_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = node_feats.shape[0]

        node_emb = self.node_encoder(node_feats)
        edge_emb = self.edge_encoder(edge_feats)

        node_layers = [node_emb]
        edge_layers = [edge_emb]
        for mp_layer, edge_updater in zip(self.mp_layers, self.edge_updaters):
            node_emb = mp_layer(node_emb, edge_emb)
            edge_emb = edge_updater(edge_emb, node_emb)
            node_layers.append(node_emb)
            edge_layers.append(edge_emb)

        node_final = self.jk_node_proj(torch.cat(node_layers, dim=-1))
        edge_final = self.jk_edge_proj(torch.cat(edge_layers, dim=-1))

        src = node_final.unsqueeze(1).expand(N, N, self.hidden_dim)
        dst = node_final.unsqueeze(0).expand(N, N, self.hidden_dim)
        edge_input = torch.cat([edge_final, src, dst], dim=-1)
        edge_logits = self.edge_classifier(edge_input).squeeze(-1)

        diag_mask = torch.eye(N, device=edge_logits.device, dtype=torch.bool)
        edge_logits = edge_logits.masked_fill(diag_mask, -6e4)

        root_logits = self.root_classifier(node_final).squeeze(-1)
        latent_logits = self.latent_head(node_final).squeeze(-1)

        edge_probs = torch.sigmoid(edge_logits)
        h_acyclicity = acyclicity_constraint(edge_probs)
        return edge_logits, root_logits, latent_logits, h_acyclicity


# ---------------------------------------------------------------------------
# Two-pass wrapper
# ---------------------------------------------------------------------------
class TwoPassGNN(nn.Module):
    """Two-pass GNN with 6 latent edge features bridging the passes.

    Pass 1 is deliberately smaller (``n_layers - 1``) and only provides edge
    logits for an auxiliary loss. Pass 2 sees ``58 + 6`` edge features and
    produces the final output (edge/root/latent logits + acyclicity).
    """

    def __init__(
        self,
        n_edge_features: int = 58,
        n_node_features: int = 15,
        n_pe_features: int = 8,
        hidden_dim: int = 128,
        n_layers: int = 5,
        n_heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.pass1 = MeterHierarchyGNN(
            n_edge_features=n_edge_features,
            n_node_features=n_node_features,
            n_pe_features=n_pe_features,
            hidden_dim=hidden_dim,
            n_layers=max(n_layers - 1, 2),
            n_heads=n_heads,
            dropout=dropout,
        )
        self.pass2 = MeterHierarchyGNN(
            n_edge_features=n_edge_features + 6,
            n_node_features=n_node_features,
            n_pe_features=n_pe_features,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(
        self, node_feats: torch.Tensor, edge_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_logits_p1, _, _, _ = self.pass1(node_feats, edge_feats)
        latent_ef = compute_latent_edge_features(edge_logits_p1)
        edge_feats_p2 = torch.cat([edge_feats, latent_ef], dim=-1)
        edge_logits, root_logits, latent_logits, h_acy = self.pass2(node_feats, edge_feats_p2)
        return edge_logits, root_logits, latent_logits, h_acy, edge_logits_p1
