"""Evaluate the GNN ensemble and/or the baselines on a dataset.

All methods share the Edmonds decoder and the edge-level F1 metric, so the
numbers are directly comparable. Baselines consume the raw ``(T, N)`` matrix;
the GNN consumes normalized features (using the statistics saved in its
checkpoint).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .model.gnn import TwoPassGNN
from .model.decode import find_best_tree
from .utils.device import get_device, device_report
from .utils.metrics import evaluate_f1
from .data.case_io import load_case, build_case_features, iter_case_dirs
from .train_gnn import apply_norm, ensemble_predict
from .baselines import get_scorer, available


def load_gnn_ensemble(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    norm = {k: np.asarray(v) for k, v in ckpt["norm"].items()}
    pe_dim = int(ckpt.get("pe_dim", 8))
    models = []
    for cfg, sd in zip(ckpt["ensemble_configs"], ckpt["ensemble_state_dicts"]):
        m = TwoPassGNN(
            n_edge_features=58, n_node_features=15, n_pe_features=pe_dim,
            hidden_dim=cfg["hidden"], n_layers=cfg["layers"],
            n_heads=cfg["heads"], dropout=cfg["dropout"],
        ).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models, norm, pe_dim


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def evaluate(
    data_dir: Path,
    split: str,
    gnn_ckpt: Optional[Path],
    baseline_names: List[str],
    device: torch.device,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    folders = iter_case_dirs(Path(data_dir) / split, limit=limit)
    raws = [r for r in (load_case(f) for f in folders) if r is not None and r["n_meters"] >= 2]
    if not raws:
        raise SystemExit(f"no cases under {data_dir}/{split}")

    results: Dict[str, List[float]] = {}

    if gnn_ckpt is not None:
        models, norm, pe_dim = load_gnn_ensemble(Path(gnn_ckpt), device)
        f1s = []
        for raw in raws:
            feat = build_case_features(raw, pe_dim=pe_dim)
            apply_norm([feat], norm)
            probs = ensemble_predict(models, feat, device)
            f1, _, _ = evaluate_f1(find_best_tree(raw["n_meters"], probs), raw["true_edges"])
            f1s.append(f1)
        results["GNN"] = f1s

    for name in baseline_names:
        scorer = get_scorer(name)
        f1s = []
        for raw in raws:
            try:
                S = scorer(raw["data"])
                f1, _, _ = evaluate_f1(find_best_tree(raw["n_meters"], S), raw["true_edges"])
            except Exception:
                f1 = 0.0
            f1s.append(f1)
        results[name] = f1s

    print(f"\nEvaluation on {data_dir}/{split}  ({len(raws)} cases)")
    print(f"{'Method':<12}{'F1 mean':>10}{'F1 median':>12}")
    print("-" * 34)
    summary = {}
    for name, f1s in results.items():
        summary[name] = _mean(f1s)
        print(f"{name:<12}{_mean(f1s):>10.4f}{float(np.median(f1s)) if f1s else 0.0:>12.4f}")
    return summary


def _parse_baselines(arg: str) -> List[str]:
    if arg.lower() == "all":
        return available()
    return [b.strip() for b in arg.split(",") if b.strip()]


def _pick_split(data_dir: Path, requested: str) -> str:
    if (Path(data_dir) / requested).exists():
        return requested
    for s in ("test", "val", "train"):
        if (Path(data_dir) / s).exists():
            return s
    return requested


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate the GNN ensemble and baselines.")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--gnn", type=str, default=None, help="GNN ensemble checkpoint")
    p.add_argument("--baselines", type=str, default="", help="comma list or 'all' (default: none)")
    p.add_argument("--avici", type=str, default=None, help="AVICI checkpoint path (sets MH_AVICI_CKPT)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "dml", "cpu"])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    if args.avici:
        os.environ["MH_AVICI_CKPT"] = args.avici
    device = get_device(args.device)
    print(device_report())
    split = _pick_split(Path(args.data), args.split)
    names = _parse_baselines(args.baselines) if args.baselines else []
    evaluate(Path(args.data), split, Path(args.gnn) if args.gnn else None, names, device, args.limit)


def main_baselines(argv: Optional[List[str]] = None) -> None:
    """Entry point for ``mh-run-baselines``: evaluate all baselines (no GNN)."""
    p = argparse.ArgumentParser(description="Evaluate all baselines on a dataset.")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--baselines", type=str, default="all")
    p.add_argument("--avici", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "dml", "cpu"])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    if args.avici:
        os.environ["MH_AVICI_CKPT"] = args.avici
    device = get_device(args.device)
    print(device_report())
    split = _pick_split(Path(args.data), args.split)
    evaluate(Path(args.data), split, None, _parse_baselines(args.baselines), device, args.limit)


if __name__ == "__main__":
    main()
