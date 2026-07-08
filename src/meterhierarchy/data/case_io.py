"""Load on-disk cases and assemble model-ready features.

A case directory contains exactly one ``*.parquet`` (consumption matrix, one
column per meter, plus a timestamp column) and one ``*.json`` (hierarchy
metadata). :func:`load_case` parses both into a raw dict; :func:`build_case_features`
turns that into the feature/label tensors consumed by the GNN.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..features import (
    compute_pair_features_fast,
    compute_node_features_fast,
    compute_laplacian_pe,
)

_SKIP_COLS = {"timestamp", "datetime", "date", "time", "zeit", "zeitstempel", "index"}
_META_KEYS = {
    "case_id", "split", "time_resolution_minutes", "n_timestamps", "n_meters",
    "n_roots", "max_depth", "topology_family", "approx_sum_children_equals_parent",
    "format_notes", "consistency_metric_mean_abs_error_over_child_sum",
}


def iter_case_dirs(split_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """Return sorted case sub-directories of ``split_dir`` (optionally capped)."""
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    folders = sorted(p for p in split_dir.iterdir() if p.is_dir())
    return folders[:limit] if limit is not None else folders


def load_case(folder: Path) -> Optional[dict]:
    """Parse one case folder into ``{data, names, true_edges, n_meters, meta, case_id}``.

    ``true_edges`` is a set of ``(parent_index, child_index)`` tuples. Supports
    both the per-meter ``meters[].parent_id`` schema and a plain
    ``{parent: [children]}`` mapping.
    """
    import pandas as pd

    folder = Path(folder)
    pq = list(folder.glob("*.parquet"))
    jp = list(folder.glob("*.json"))
    if not pq or not jp:
        return None

    df = pd.read_parquet(pq[0])
    cols = [c for c in df.columns if c.lower().strip() not in _SKIP_COLS]
    names = [c.strip().strip('"').replace("﻿", "") for c in cols]
    data = df[cols].values.astype(np.float64)
    data = np.nan_to_num(data, nan=0.0)

    payload = json.loads(jp[0].read_text(encoding="utf-8"))
    index = {n: i for i, n in enumerate(names)}
    true_edges = set()

    payload_edges = payload.get("parent_to_children", payload) if isinstance(payload, dict) else payload

    if isinstance(payload_edges, dict) and isinstance(payload_edges.get("meters"), list):
        for m in payload_edges["meters"]:
            mid = m["id"].strip().strip('"')
            pid = m.get("parent_id")
            if pid:
                pc = pid.strip().strip('"')
                if pc in index and mid in index:
                    true_edges.add((index[pc], index[mid]))
    elif isinstance(payload_edges, dict):
        for parent, children in payload_edges.items():
            if parent in _META_KEYS or not isinstance(children, list):
                continue
            pc = parent.strip().strip('"')
            if pc not in index:
                continue
            for child in children:
                cc = child.strip().strip('"')
                if cc in index:
                    true_edges.add((index[pc], index[cc]))

    meta: Dict = {}
    if isinstance(payload, dict):
        for k in ("n_roots", "max_depth", "approx_sum_children_equals_parent"):
            if k in payload:
                meta[k] = payload[k]

    return {
        "data": data,
        "names": names,
        "true_edges": true_edges,
        "n_meters": len(names),
        "meta": meta,
        "case_id": folder.name,
    }


def build_case_features(raw: dict, pe_dim: int = 8) -> dict:
    """Assemble GNN-ready features + labels from a raw case (see :func:`load_case`).

    Returns a dict with ``edge_feats (N,N,58)``, ``node_feats (N,15)``,
    ``pe (N,pe_dim)``, ``labels (N,N)``, ``root_labels (N,)``,
    ``latent_labels (N,)``, ``n_meters``, ``is_additive``, plus ``true_edges``
    and ``case_id`` passthrough.
    """
    data = raw["data"]
    edges_set = raw["true_edges"]
    meta = raw.get("meta", {})
    N = data.shape[1]

    feat_cube, _abs_means, corr_raw = compute_pair_features_fast(data)
    node_feats = compute_node_features_fast(data, corr_raw)
    pe = compute_laplacian_pe(data, k=pe_dim)

    labels = np.zeros((N, N), dtype=np.float32)
    for p, c in edges_set:
        labels[p, c] = 1.0

    all_children = {c for _, c in edges_set}
    root_labels = np.zeros(N, dtype=np.float32)
    for i in range(N):
        if i not in all_children:
            root_labels[i] = 1.0

    is_additive = bool(meta.get("approx_sum_children_equals_parent", True))
    latent_labels = np.zeros(N, dtype=np.float32)
    if not is_additive:
        for p, _c in edges_set:
            latent_labels[p] = 1.0

    return {
        "edge_feats": feat_cube.astype(np.float32),
        "node_feats": node_feats.astype(np.float32),
        "pe": pe.astype(np.float32),
        "labels": labels,
        "root_labels": root_labels,
        "latent_labels": latent_labels,
        "n_meters": N,
        "is_additive": is_additive,
        "true_edges": edges_set,
        "case_id": raw.get("case_id", "case"),
    }
