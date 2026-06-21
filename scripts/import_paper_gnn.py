#!/usr/bin/env python3
"""Combine trained TwoPassGNN run-ensembles into one repo-format checkpoint.

Use this to evaluate an externally trained GNN (e.g. the paper's per-seed
``GNNv2_ensemble.pt`` files) through this repository's pipeline. Each input is a
checkpoint with keys ``ensemble_state_dicts``, ``ensemble_configs``,
``edge_mean``/``edge_std``/``node_mean``/``node_std``/``pe_mean``/``pe_std`` and
``pe_dim``. The output is the repo format
``{ensemble_state_dicts, ensemble_configs, norm: {...}, pe_dim}`` with the
normalization statistics averaged across the inputs (they are near-identical
across seeds drawn from the same synthetic generator).

Example:
  python scripts/import_paper_gnn.py "<...>/runs/Housing/RUN_0*/GNNv2_ensemble.pt" \
      --out checkpoints/gnn_ensemble.pt
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import torch

_NORM_KEYS = ["edge_mean", "edge_std", "node_mean", "node_std", "pe_mean", "pe_std"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Combine trained GNN run-ensembles into one repo checkpoint.")
    p.add_argument("inputs", nargs="+", help="run ensemble .pt files (globs allowed)")
    p.add_argument("--out", required=True, help="output checkpoint path")
    args = p.parse_args(argv)

    files = []
    for pattern in args.inputs:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        raise SystemExit("no input checkpoints matched")

    state_dicts, configs, norms = [], [], []
    pe_dim = 8
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        state_dicts.extend(ck["ensemble_state_dicts"])
        configs.extend(ck["ensemble_configs"])
        norms.append({k: np.asarray(ck[k], dtype=np.float64) for k in _NORM_KEYS})
        pe_dim = int(ck.get("pe_dim", pe_dim))

    norm = {k: np.mean([n[k] for n in norms], axis=0).astype(np.float32) for k in _NORM_KEYS}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"ensemble_state_dicts": state_dicts, "ensemble_configs": configs, "norm": norm, "pe_dim": pe_dim},
        out,
    )
    print(f"combined {len(state_dicts)} models from {len(files)} file(s) -> {out}")


if __name__ == "__main__":
    main()
