"""Load the paper's *real* sub-metering datasets into the same per-building
dict shape produced by :mod:`meterhierarchy.data.case_io` for synthetic cases.

Each real dataset lives in a folder containing two files:

* ``consolidated.csv`` -- a wide table with one time-like column (not a meter)
  and one numeric column per meter. Meter columns are named by meter id.
* ``hierarchy.json`` -- the parent/child topology, in one of several shapes.

The public entry point is :func:`load_real_buildings`, which returns one dict
per building::

    {
        "dataset":    str,                 # short dataset name (e.g. "REDD")
        "building":   str,                 # building id within the dataset
        "data":       np.ndarray (T, n),   # float, NaN -> 0
        "names":      list[str],           # meter ids, local index 0..n-1
        "true_edges": set[(parent_local_idx, child_local_idx)],
    }

The returned ``data`` / ``names`` / ``true_edges`` triplet matches exactly what
:func:`case_io.load_case` yields, so real buildings can be fed straight into the
same edge scorers as the synthetic data.

Only the standard library, numpy and pandas are used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Dataset registry: short name -> on-disk folder name under the data root.
# --------------------------------------------------------------------------- #
DATASETS: Dict[str, str] = {
    "AMPds2": "07_AMPds2",
    "REFIT": "06_REFIT",
    "REDD": "19_REDD",
    "UKDALE": "18_UKDALE",
    "RAE": "09_RAE",
    "PRECON": "08_PRECON",
}

# Maximum number of (evenly spaced) rows kept when the CSV is large and the
# caller did not cap it via ``max_rows``.
_SUBSAMPLE_CAP = 5000

# Column-name fragments that identify the (non-meter) time axis. Matched case
# insensitively; ``timestamp_houseN`` columns (REFIT) are caught by the prefix
# check in :func:`_is_time_column`.
_TIME_NAMES = {"timestamp", "index", "datetime", "date", "time"}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _is_time_column(col: str) -> bool:
    """True if ``col`` is a time axis rather than a meter."""
    low = col.strip().strip('"').lower()
    if low in _TIME_NAMES:
        return True
    # REFIT uses one timestamp column per house: "timestamp_house1", ...
    if low.startswith("timestamp_") or low.startswith("time_") or low.startswith("date_"):
        return True
    return False


def _clean_id(name: str) -> str:
    """Normalise a column / hierarchy id (strip quotes and BOM)."""
    return str(name).strip().strip('"').replace("﻿", "")


def _building_of(meter_id: str) -> Optional[str]:
    """Return the building prefix (``house1`` / ``bldg1`` ...) of a meter id.

    Buildings are identified by the token before the first underscore when that
    token looks like ``house<NN>`` or ``bldg<NN>``. Returns ``None`` when the id
    carries no such prefix (e.g. AMPds2 meters like ``WHE_P``), in which case the
    caller treats the whole dataset as a single building.
    """
    head = meter_id.split("_", 1)[0]
    low = head.lower()
    for tag in ("house", "bldg"):
        if low.startswith(tag) and low[len(tag):].isdigit():
            return head
    return None


def _is_meter_node(value: object) -> bool:
    """True if a hierarchy value describes a meter node (has a ``children`` key).

    Used to distinguish real meter entries from metadata blobs that happen to
    live alongside them (e.g. RAE's ``paired_240v_circuits``).
    """
    return isinstance(value, dict) and "children" in value


def _children_ids(node: dict) -> List[str]:
    """Extract child meter ids from a node, supporting list- and dict-children."""
    children = node.get("children", [])
    if isinstance(children, dict):
        return [_clean_id(k) for k in children.keys()]
    if isinstance(children, list):
        return [_clean_id(c) for c in children]
    return []


# --------------------------------------------------------------------------- #
# Hierarchy parsing -> flat {parent_id: [child_id, ...]} edge map
# --------------------------------------------------------------------------- #
def _flatten_hierarchy(hierarchy: dict) -> Dict[str, List[str]]:
    """Flatten a ``hierarchy`` object into ``{parent_id: [child_id, ...]}``.

    Handles three shapes uniformly:

    * FLAT (AMPds2, REDD, UKDALE, REFIT-explicit): top-level keys are meter ids
      whose ``children`` is a list.
    * FLAT with dict-children (PRECON): ``children`` is a ``{child_id: {...}}``
      mapping.
    * NESTED by building (RAE): top-level keys are building ids whose *values*
      are dicts of meter nodes; we recurse one level into those.

    Non-meter metadata entries (dicts without a ``children`` key) are skipped at
    every level.
    """
    edges: Dict[str, List[str]] = {}

    def add_node(node_id: str, node: dict) -> None:
        kids = _children_ids(node)
        if kids:
            edges.setdefault(node_id, []).extend(kids)

    for key, value in hierarchy.items():
        if _is_meter_node(value):
            # Flat layout: this key is itself a meter node.
            add_node(_clean_id(key), value)
        elif isinstance(value, dict):
            # Either a nested per-building dict of meter nodes (RAE), or an
            # unrelated metadata blob. Recurse one level and keep only the
            # entries that are genuine meter nodes.
            for sub_id, sub_node in value.items():
                if _is_meter_node(sub_node):
                    add_node(_clean_id(sub_id), sub_node)
    return edges


def _implicit_refit_edges(payload: dict) -> Dict[str, List[str]]:
    """Build the implicit REFIT tree from ``appliance_labels`` when no
    ``hierarchy`` key is present.

    For each ``houseK`` the root is ``houseK_Aggregate`` and its children are
    ``houseK_<ApplianceName>`` for every appliance label key listed.
    """
    labels = payload.get("appliance_labels", {})
    edges: Dict[str, List[str]] = {}
    for house, appliances in labels.items():
        root = f"{house}_Aggregate"
        edges[root] = [f"{house}_{appliance}" for appliance in appliances.keys()]
    return edges


def _load_edge_map(payload: dict) -> Dict[str, List[str]]:
    """Return the dataset's full ``{parent_id: [child_id, ...]}`` edge map."""
    hierarchy = payload.get("hierarchy")
    if isinstance(hierarchy, dict) and hierarchy:
        return _flatten_hierarchy(hierarchy)
    # REFIT-style fallback: no explicit hierarchy, derive it from labels.
    return _implicit_refit_edges(payload)


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #
def _read_consolidated(csv_path: Path, max_rows: Optional[int]) -> pd.DataFrame:
    """Read ``consolidated.csv`` and return a meter-only float DataFrame.

    Time-like columns are dropped, every remaining column is coerced to float
    with NaNs filled as 0, and the row count is reduced (head via ``nrows`` when
    ``max_rows`` is given, otherwise evenly-spaced subsampling to
    :data:`_SUBSAMPLE_CAP`).

    AMPds2 special case: meters come as ``<id>_P`` (active power) and ``<id>_Pt``
    (energy) pairs. Only the ``_P`` columns are kept because the hierarchy ids
    already end in ``_P``; the ``_Pt`` columns are dropped.
    """
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)

    # Drop the time axis / axes.
    meter_cols = [c for c in df.columns if not _is_time_column(c)]
    df = df[meter_cols]

    # AMPds2: keep only the active-power (_P) channels, discard energy (_Pt).
    if any(_clean_id(c).endswith("_Pt") for c in df.columns):
        df = df[[c for c in df.columns if not _clean_id(c).endswith("_Pt")]]

    # Coerce to numeric float; KEEP NaN so per-building recording windows can be
    # detected downstream (real datasets store buildings in disjoint windows).
    df = df.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    df.columns = [_clean_id(c) for c in df.columns]
    return df


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_real_buildings(dataset_dir: Path, max_rows: Optional[int] = None) -> List[dict]:
    """Load every building from a single real-dataset folder.

    Parameters
    ----------
    dataset_dir:
        Folder containing ``consolidated.csv`` and ``hierarchy.json``.
    max_rows:
        If given, only the first ``max_rows`` CSV rows are read (fast path via
        pandas ``nrows``). If ``None`` and the series is long, it is subsampled
        to at most :data:`_SUBSAMPLE_CAP` evenly spaced rows.

    Returns
    -------
    list of dict
        One dict per building, with keys ``dataset``, ``building``, ``data``,
        ``names`` and ``true_edges`` (see module docstring). Buildings with
        fewer than two present meters or no edges are skipped.
    """
    dataset_dir = Path(dataset_dir)
    payload = json.loads((dataset_dir / "hierarchy.json").read_text(encoding="utf-8"))
    dataset_name = _clean_id(payload.get("dataset", dataset_dir.name))

    edge_map = _load_edge_map(payload)
    df = _read_consolidated(dataset_dir / "consolidated.csv", max_rows)
    present_cols = list(df.columns)
    present = set(present_cols)

    # Collect every meter id that participates in the hierarchy (parents +
    # children). The CSV is the source of truth for which actually exist.
    hierarchy_ids: Set[str] = set(edge_map.keys())
    for kids in edge_map.values():
        hierarchy_ids.update(kids)

    # Group meter ids into buildings by their id prefix. Meters that carry no
    # building prefix (AMPds2) all fall into a single building named "house".
    building_to_meters: Dict[str, List[str]] = {}
    for meter in present_cols:
        if meter not in hierarchy_ids:
            continue  # CSV column not referenced by the hierarchy: ignore.
        bld = _building_of(meter) or "house"
        building_to_meters.setdefault(bld, []).append(meter)

    buildings: List[dict] = []
    for building in sorted(building_to_meters, key=_building_sort_key):
        # Candidate meters of this building that exist as CSV columns, in CSV
        # order (for deterministic local indices).
        candidates = [m for m in present_cols if m in building_to_meters[building]]
        sub = df[candidates]

        # Restrict to this building's recording window: drop rows where none of
        # its meters has a reading. Then drop meters that never record in it.
        sub = sub.dropna(how="all")
        local_names = [c for c in candidates if bool(sub[c].notna().any())]
        if len(local_names) < 2:
            continue
        sub = sub[local_names].fillna(0.0)

        # Evenly subsample very long windows.
        if len(sub) > _SUBSAMPLE_CAP:
            idx = np.unique(np.linspace(0, len(sub) - 1, _SUBSAMPLE_CAP).round().astype(int))
            sub = sub.iloc[idx]

        index = {name: i for i, name in enumerate(local_names)}
        true_edges: Set[Tuple[int, int]] = set()
        for parent, kids in edge_map.items():
            if parent not in index:
                continue
            for child in kids:
                if child in index:
                    true_edges.add((index[parent], index[child]))

        # Skip degenerate buildings (need >= 2 meters, an edge, and some data).
        if not true_edges or len(sub) < 10:
            continue

        data = np.nan_to_num(sub.to_numpy(dtype=np.float64), nan=0.0)
        buildings.append(
            {
                "dataset": dataset_name,
                "building": building,
                "data": data,
                "names": local_names,
                "true_edges": true_edges,
            }
        )

    return buildings


def _building_sort_key(name: str):
    """Sort buildings naturally: ``house2`` before ``house10``."""
    for tag in ("house", "bldg"):
        if name.lower().startswith(tag) and name[len(tag):].isdigit():
            return (0, tag, int(name[len(tag):]))
    return (1, name, 0)


def load_real_dataset(root: Path, name: str, max_rows: Optional[int] = None) -> List[dict]:
    """Load a real dataset by its short ``name`` (resolved via :data:`DATASETS`).

    Parameters
    ----------
    root:
        The ``RealDataClean`` directory holding the numbered dataset folders.
    name:
        Short dataset key, e.g. ``"REDD"`` (see :data:`DATASETS`).
    max_rows:
        Forwarded to :func:`load_real_buildings`.
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; known: {sorted(DATASETS)}")
    return load_real_buildings(Path(root) / DATASETS[name], max_rows=max_rows)


def apply_apriori_filter(name: str, buildings: List[dict]) -> List[dict]:
    """Apply paper-specific a-priori building filters.

    * RAE: keep only building ``house1`` (house2's mains are sub-panel feeds,
      not a clean whole-house aggregate).
    * UKDALE: drop building ``bldg1`` (excluded a priori).
    * Any other dataset: returned unchanged.
    """
    if name == "RAE":
        return [b for b in buildings if b["building"] == "house1"]
    if name == "UKDALE":
        return [b for b in buildings if b["building"] != "bldg1"]
    return buildings
