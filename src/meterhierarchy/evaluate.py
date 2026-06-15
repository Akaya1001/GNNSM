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


def evaluate_real(
    data_root: Path,
    dataset_names: List[str],
    gnn_ckpt: Optional[Path],
    baseline_names: List[str],
    device: torch.device,
    max_rows: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate methods on the paper's real datasets (mean F1 over buildings).

    Applies the a-priori rooted-tree filter (RAE house1 only; UK-DALE drops
    building 1) before scoring.
    """
    from .data.real_io import load_real_dataset, apply_apriori_filter

    models = norm = None
    pe_dim = 8
    if gnn_ckpt is not None:
        models, norm, pe_dim = load_gnn_ensemble(Path(gnn_ckpt), device)

    methods = (["GNN"] if models else []) + list(baseline_names)
    table: Dict[str, Dict[str, float]] = {m: {} for m in methods}

    for name in dataset_names:
        try:
            buildings = apply_apriori_filter(name, load_real_dataset(Path(data_root), name, max_rows=max_rows))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip {name}] {exc}")
            continue
        for m in methods:
            f1s = []
            for b in buildings:
                N = b["data"].shape[1]
                try:
                    if m == "GNN":
                        feat = build_case_features(
                            {"data": b["data"], "true_edges": b["true_edges"], "meta": {}, "case_id": b["building"]},
                            pe_dim=pe_dim,
                        )
                        apply_norm([feat], norm)
                        S = ensemble_predict(models, feat, device)
                    else:
                        S = get_scorer(m)(b["data"])
                    f1, _, _ = evaluate_f1(find_best_tree(N, S), b["true_edges"])
                except Exception:  # noqa: BLE001
                    f1 = 0.0
                f1s.append(f1)
            table[m][name] = _mean(f1s)

    header = "Method".ljust(11) + "".join(f"{n:>9}" for n in dataset_names) + f"{'MEAN':>9}"
    print(f"\nReal-data evaluation (mean F1 over buildings; max_rows={max_rows})")
    print(header)
    print("-" * len(header))
    for m in methods:
        vals = [table[m].get(n, 0.0) for n in dataset_names]
        mean = sum(vals) / len(vals) if vals else 0.0
        print(m.ljust(11) + "".join(f"{v:>9.3f}" for v in vals) + f"{mean:>9.3f}")
    return table


def main_real(argv: Optional[List[str]] = None) -> None:
    """Entry point for ``mh-evaluate-real``."""
    p = argparse.ArgumentParser(description="Evaluate GNN/baselines on the paper's real datasets.")
    p.add_argument("--data-root", type=str, required=True, help="path to RealDataClean/")
    p.add_argument("--datasets", type=str, default="AMPds2,REFIT,REDD,UKDALE,RAE")
    p.add_argument("--gnn", type=str, default=None)
    p.add_argument("--baselines", type=str, default="", help="comma list or 'all'")
    p.add_argument("--avici", type=str, default=None, help="AVICI checkpoint (sets MH_AVICI_CKPT)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "dml", "cpu"])
    p.add_argument("--max-rows", type=int, default=4000, help="rows read per dataset (None=subsample 5000)")
    args = p.parse_args(argv)
    if args.avici:
        os.environ["MH_AVICI_CKPT"] = args.avici
    device = get_device(args.device)
    print(device_report())
    names = _parse_baselines(args.baselines) if args.baselines else []
    ds = [d.strip() for d in args.datasets.split(",") if d.strip()]
    mr = None if args.max_rows is not None and args.max_rows < 0 else args.max_rows
    evaluate_real(Path(args.data_root), ds, Path(args.gnn) if args.gnn else None, names, device, mr)


if __name__ == "__main__":
    main()
