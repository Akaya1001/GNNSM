"""Apply a trained GNN ensemble to a raw meter CSV and output the predicted hierarchy.

Inference only, no ground truth required. Each CSV must have one time-like column
(dropped automatically) and one numeric column per meter, the column header being
the meter id. Constant or empty meter columns are discarded, mirroring the paper's
preprocessing. The model was trained on 15-min data; markedly different sampling
rates may reduce accuracy (see RUN_PRETRAINED.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .data.case_io import build_case_features
from .data.real_io import _is_time_column, _clean_id
from .model.decode import find_best_tree
from .train_gnn import apply_norm, ensemble_predict
from .evaluate import load_gnn_ensemble
from .utils.device import get_device, device_report


def load_meter_csv(csv_path: Path, max_rows: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    """Read a meter CSV into a ``(T, N)`` matrix and the surviving meter names."""
    df = pd.read_csv(csv_path, nrows=max_rows)
    meter_cols = [c for c in df.columns if not _is_time_column(str(c))]
    if not meter_cols:
        raise SystemExit(f"{csv_path}: no meter columns found (all look time-like)")
    x = df[meter_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    names = [_clean_id(str(c)) for c in meter_cols]
    # Drop constant / all-NaN meters (the paper discards constant sensors).
    keep = [j for j in range(x.shape[1])
            if np.isfinite(x[:, j]).sum() >= 2 and np.nanstd(x[:, j]) > 0]
    if len(keep) < 2:
        raise SystemExit(f"{csv_path}: fewer than 2 non-constant meters after cleaning")
    return np.nan_to_num(x[:, keep], nan=0.0), [names[j] for j in keep]


def predict_tree(models, norm, pe_dim: int, data: np.ndarray, device) -> dict:
    """Return ``{child_idx: parent_idx}`` for the ``(T, N)`` matrix ``data``."""
    feat = build_case_features({"data": data, "true_edges": set(), "meta": {}}, pe_dim=pe_dim)
    apply_norm([feat], norm)
    probs = ensemble_predict(models, feat, device)
    return find_best_tree(data.shape[1], probs)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Predict a meter hierarchy from CSV(s) with a trained GNN ensemble.")
    p.add_argument("--gnn", required=True, help="path to the trained ensemble (gnn_ensemble.pt)")
    p.add_argument("--csv", nargs="+", required=True, help="one or more meter CSV files")
    p.add_argument("--out", default=None, help="write predictions as JSON to this path")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "dml", "cpu"])
    p.add_argument("--max-rows", type=int, default=None, help="read at most this many rows per CSV")
    args = p.parse_args(argv)

    device = get_device(args.device)
    print(device_report())
    models, norm, pe_dim = load_gnn_ensemble(Path(args.gnn), device)

    results = {}
    for csv in args.csv:
        data, names = load_meter_csv(Path(csv), args.max_rows)
        parent_of = predict_tree(models, norm, pe_dim, data, device)
        children = set(parent_of)
        root = next((i for i in range(len(names)) if i not in children), 0)
        edges = [(names[pa], names[ch]) for ch, pa in sorted(parent_of.items())]
        print(f"\n{csv}: {len(names)} meters, {data.shape[0]} timesteps")
        print(f"  ROOT: {names[root]}")
        for pa, ch in edges:
            print(f"  {ch}  <-  {pa}")
        results[str(csv)] = {
            "root": names[root],
            "meters": names,
            "edges_parent_child": [[pa, ch] for pa, ch in edges],
        }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
