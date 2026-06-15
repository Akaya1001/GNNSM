"""Train the TwoPassGNN ensemble for meter hierarchy detection.

Pipeline: load cases -> build features -> z-normalize (edge/node/pe with train
statistics) -> train each ensemble member -> save a checkpoint bundling the
member weights, their configs, and the normalization statistics (so evaluation
applies the *same* normalization to fresh data).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .model.gnn import TwoPassGNN
from .model.decode import find_best_tree
from .utils.device import get_device, device_report
from .utils.metrics import evaluate_f1
from .data.case_io import load_case, build_case_features, iter_case_dirs

# Default ensemble (the A/C configuration from the paper).
DEFAULT_ENSEMBLE = [
    {"name": "A", "hidden": 256, "layers": 5, "heads": 8, "dropout": 0.15, "lr": 1.5e-4},
    {"name": "C", "hidden": 192, "layers": 6, "heads": 8, "dropout": 0.10, "lr": 1.3e-4},
]
PE_DIM = 8
LAMBDA_ACY_MAX = 1.0
LAMBDA_ROOT = 0.5
LAMBDA_LATENT = 0.3
AUX_WEIGHT = 0.3
NONADD_CASE_WEIGHT = 2.0
GRAD_ACCUM_STEPS = 4


# --------------------------------------------------------------------------
# Data loading + normalization
# --------------------------------------------------------------------------
def load_split(data_dir: Path, split: str, limit: Optional[int], pe_dim: int = PE_DIM) -> List[dict]:
    cases = []
    for folder in iter_case_dirs(Path(data_dir) / split, limit=limit):
        raw = load_case(folder)
        if raw is None or raw["n_meters"] < 2:
            continue
        cases.append(build_case_features(raw, pe_dim=pe_dim))
    return cases


def compute_norm(train_cases: List[dict]) -> Dict[str, np.ndarray]:
    edge_chunks, node_all, pe_all = [], [], []
    for c in train_cases:
        N = c["n_meters"]
        off_diag = ~np.eye(N, dtype=bool)
        edge_chunks.append(c["edge_feats"][off_diag].reshape(-1, c["edge_feats"].shape[-1]))
        node_all.append(c["node_feats"])
        pe_all.append(c["pe"])
    edge_stack = np.concatenate(edge_chunks, axis=0)
    edge_mean = edge_stack.mean(axis=0).astype(np.float32)
    edge_std = edge_stack.std(axis=0).astype(np.float32)
    edge_std[edge_std < 1e-6] = 1.0
    node_cat = np.concatenate(node_all)
    node_mean = node_cat.mean(axis=0).astype(np.float32)
    node_std = node_cat.std(axis=0).astype(np.float32)
    node_std[node_std < 1e-6] = 1.0
    pe_cat = np.concatenate(pe_all)
    pe_mean = pe_cat.mean(axis=0).astype(np.float32)
    pe_std = pe_cat.std(axis=0).astype(np.float32)
    pe_std[pe_std < 1e-6] = 1.0
    return {
        "edge_mean": edge_mean, "edge_std": edge_std,
        "node_mean": node_mean, "node_std": node_std,
        "pe_mean": pe_mean, "pe_std": pe_std,
    }


def apply_norm(cases: List[dict], norm: Dict[str, np.ndarray]) -> None:
    """In-place: set normalized ``edge_feats``/``node_feats`` and add ``pe_norm``."""
    for c in cases:
        c["edge_feats"] = np.nan_to_num(
            ((c["edge_feats"] - norm["edge_mean"]) / norm["edge_std"]).astype(np.float32),
            nan=0, posinf=5, neginf=-5,
        )
        c["node_feats"] = np.nan_to_num(
            ((c["node_feats"] - norm["node_mean"]) / norm["node_std"]).astype(np.float32),
            nan=0, posinf=5, neginf=-5,
        )
        c["pe_norm"] = np.nan_to_num(
            ((c["pe"] - norm["pe_mean"]) / norm["pe_std"]).astype(np.float32),
            nan=0, posinf=5, neginf=-5,
        )


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def balanced_indices(cases: List[dict], rng: np.random.Generator) -> List[int]:
    """Interleave additive/non-additive cases roughly 50/50."""
    add_idx = [i for i, c in enumerate(cases) if c["is_additive"]]
    non_idx = [i for i, c in enumerate(cases) if not c["is_additive"]]
    rng.shuffle(add_idx)
    rng.shuffle(non_idx)
    if not add_idx:
        return non_idx
    if not non_idx:
        return add_idx
    out: List[int] = []
    n_a, n_n = len(add_idx), len(non_idx)
    for k in range(max(n_a, n_n)):
        out.append(add_idx[k % n_a])
        out.append(non_idx[k % n_n])
    return out


def train_one_model(
    name: str,
    config: dict,
    splits: dict,
    pe_dim: int,
    device: torch.device,
    log: Callable[[str], None],
    n_epochs: int,
    patience: int,
    lambda_root: float = LAMBDA_ROOT,
    lambda_latent: float = LAMBDA_LATENT,
    lambda_acy_max: float = LAMBDA_ACY_MAX,
    aux_weight: float = AUX_WEIGHT,
    nonadd_case_weight: float = NONADD_CASE_WEIGHT,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
) -> Tuple[TwoPassGNN, float]:
    """Train a single TwoPassGNN; return ``(best_model, best_val_f1)``."""
    model = TwoPassGNN(
        n_edge_features=58, n_node_features=15, n_pe_features=pe_dim,
        hidden_dim=config["hidden"], n_layers=config["layers"],
        n_heads=config["heads"], dropout=config["dropout"],
    ).to(device)
    log(f"  [{name}] params={sum(p.numel() for p in model.parameters()):,d}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    warmup_epochs = min(10, max(1, n_epochs // 20))
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs - warmup_epochs, 1), eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    root_criterion = nn.BCEWithLogitsLoss(reduction="mean")
    latent_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=device), reduction="mean")

    best_f1, best_state, patience_ctr = 0.0, None, 0
    rng = np.random.default_rng(42 + abs(hash(name)) % 10000)

    for epoch in range(n_epochs):
        model.train()
        if epoch < warmup_epochs:
            lambda_acy = 0.0
        else:
            lambda_acy = lambda_acy_max * min(1.0, (epoch - warmup_epochs) / max(1, warmup_epochs))

        optimizer.zero_grad(set_to_none=True)
        accum = 0
        for idx in balanced_indices(splits["train"], rng):
            c = splits["train"][idx]
            N = c["n_meters"]
            nf = np.concatenate([c["node_feats"], c["pe_norm"]], axis=1)
            nf_t = torch.from_numpy(nf).to(device)
            ef_t = torch.from_numpy(c["edge_feats"]).to(device)
            labels_t = torch.from_numpy(c["labels"]).to(device)
            root_t = torch.from_numpy(c["root_labels"]).to(device)
            latent_t = torch.from_numpy(c["latent_labels"]).to(device)

            edge_logits, root_logits, latent_logits, h_acy, edge_logits_p1 = model(nf_t, ef_t)

            n_pos = max(int(c["labels"].sum()), 1)
            n_neg = max(N * (N - 1) - n_pos, 1)
            pw = torch.tensor([n_neg / n_pos], device=device)
            edge_crit = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
            mask = ~torch.eye(N, device=device, dtype=torch.bool)
            edge_loss = edge_crit(edge_logits, labels_t)[mask].mean()
            aux_loss = edge_crit(edge_logits_p1, labels_t)[mask].mean() * aux_weight
            root_loss = root_criterion(root_logits, root_t)
            latent_loss = latent_criterion(latent_logits, latent_t)
            acy_loss = lambda_acy * h_acy
            case_weight = 1.0 if c["is_additive"] else nonadd_case_weight

            loss = (
                edge_loss * case_weight + aux_loss * case_weight
                + lambda_root * root_loss + lambda_latent * latent_loss + acy_loss
            ) / grad_accum_steps

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                continue
            loss.backward()
            accum += 1
            if accum >= grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
        if accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        scheduler.step()

        # validation
        model.eval()
        val_f1s = []
        with torch.no_grad():
            for c in splits.get("val", []):
                N = c["n_meters"]
                nf = np.concatenate([c["node_feats"], c["pe_norm"]], axis=1)
                el, _, _, _, _ = model(torch.from_numpy(nf).to(device), torch.from_numpy(c["edge_feats"]).to(device))
                probs = torch.sigmoid(el).cpu().numpy()
                f1, _, _ = evaluate_f1(find_best_tree(N, probs), c["true_edges"])
                val_f1s.append(f1)
        val_f1 = float(np.mean(val_f1s)) if val_f1s else 0.0

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"  [{name}] epoch {epoch + 1}/{n_epochs} val_f1={val_f1:.4f} (best {best_f1:.4f})")
        if patience_ctr >= patience:
            log(f"  [{name}] early stop at epoch {epoch + 1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1


@torch.no_grad()
def ensemble_predict(models: List[TwoPassGNN], case: dict, device: torch.device) -> np.ndarray:
    """Average ``sigmoid(edge_logits)`` over all ensemble members."""
    N = case["n_meters"]
    nf = np.concatenate([case["node_feats"], case["pe_norm"]], axis=1)
    nf_t = torch.from_numpy(nf).to(device)
    ef_t = torch.from_numpy(case["edge_feats"]).to(device)
    acc = np.zeros((N, N), dtype=np.float64)
    for model in models:
        model.eval()
        el, _, _, _, _ = model(nf_t, ef_t)
        acc += torch.sigmoid(el).cpu().numpy().astype(np.float64)
    return acc / max(len(models), 1)


def save_ensemble(path: Path, models: List[TwoPassGNN], configs: List[dict], norm: Dict, pe_dim: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ensemble_state_dicts": [m.state_dict() for m in models],
            "ensemble_configs": configs,
            "norm": {k: np.asarray(v) for k, v in norm.items()},
            "pe_dim": pe_dim,
        },
        path,
    )


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Train the TwoPassGNN ensemble.")
    p.add_argument("--data", type=str, default=None, help="dataset root with train/val[/test] splits")
    p.add_argument("--out", type=str, default="checkpoints/gnn_ensemble.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "dml", "cpu"])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    device = get_device(args.device)
    print(device_report())

    if args.smoke:
        from .avici.train import make_smoke_cases

        raws = make_smoke_cases(k=6, N=6, T=128, seed=0)
        cases = [build_case_features({"data": r["data"], "true_edges": r["true_edges"], "meta": {}}) for r in raws]
        splits = {"train": cases[:4], "val": cases[4:]}
        norm = compute_norm(splits["train"])
        apply_norm(splits["train"], norm)
        apply_norm(splits["val"], norm)
        cfg = {"name": "S", "hidden": 32, "layers": 3, "heads": 4, "dropout": 0.1, "lr": 2e-3}
        model, f1 = train_one_model("S", cfg, splits, PE_DIM, device, print, n_epochs=5, patience=5)
        probs = ensemble_predict([model], splits["val"][0], device)
        assert probs.shape == (splits["val"][0]["n_meters"],) * 2
        print(f"[gnn-smoke] OK (val_f1={f1:.3f}, probs {probs.shape})")
        return

    if not args.data:
        raise SystemExit("--data is required (or use --smoke).")
    splits = {}
    for split in ("train", "val", "test"):
        cs = load_split(Path(args.data), split, args.limit)
        if cs:
            splits[split] = cs
        print(f"  {split}: {len(cs)} cases")
    if "train" not in splits:
        raise SystemExit("no training cases found")
    norm = compute_norm(splits["train"])
    for cs in splits.values():
        apply_norm(cs, norm)

    models, configs = [], DEFAULT_ENSEMBLE
    t0 = time.perf_counter()
    for cfg in configs:
        model, val_f1 = train_one_model(
            cfg["name"], cfg, splits, PE_DIM, device, print, n_epochs=args.epochs, patience=args.patience
        )
        models.append(model)
        print(f"  model {cfg['name']}: val_f1={val_f1:.4f}")
    save_ensemble(Path(args.out), models, configs, norm, PE_DIM)
    print(f"saved ensemble -> {args.out}  ({(time.perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
