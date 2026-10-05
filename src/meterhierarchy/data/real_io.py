"""Load the paper's *real* sub-metering datasets into the same per-building
dict shape produced by :mod:`meterhierarchy.data.case_io` for synthetic cases.

Each real dataset lives in a folder containing two files:

* ``consolidated.csv`` -- a wide table with one time-like column (not a meter)
  and one numeric column per meter. Meter columns are named by meter id.
* ``hierarchy.json`` -- the parent/child topology, in one of several shapes.

The public entry point is :func:`load_real_dataset` (by short dataset name);
:func:`load_real_buildings` is the generic loader for a dataset folder. Both
return one dict per building::

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

The seven datasets of the paper's real-world benchmark (:data:`BENCHMARK`) are
loaded by dataset-specific parsers that reproduce the paper's protocol (Sec. 5.1.2,
Table 11): one case per building, rooted at its physical whole-house meter with
at least three sub-meters (N >= 4); series longer than :data:`TARGET_T` samples
are evenly subsampled; constant sensors are dropped. Dataset rules:

* UK-DALE: all five buildings; the duplicate whole-house (sound-card) meter that
  the metadata lists as a child of the mains in buildings 1, 2 and 5 is removed.
* UCI Power: root ``Global_active_power`` (kW -> W), children
  ``Sub_metering_1..3`` (Wh per minute -> W); the computed ``remainder`` is not
  a meter and is not used.
* Plegma: houses with at least three sub-meters whose mean sum does not exceed
  the mean aggregate (houses 1, 3, 4, 7 and 11).
* RAE is not part of the benchmark: its mains is computed as the sum of its
  circuits (``house1.txt``: "Sub-meter Mains: calc").

Other datasets fall back to the generic hierarchy-driven loader
:func:`load_real_buildings`.

Only the standard library, numpy and pandas are used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import AbstractSet, Dict, List, Optional, Set, Tuple

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
    "PRECON": "08_PRECON",
    "UCIPower": "05_UCI_Power",
    "Plegma": "10_Plegma",
    "RAE": "09_RAE",  # loadable, but excluded from the benchmark (computed mains)
}

# The paper's real-world benchmark (Table 11), in table order.
BENCHMARK: List[str] = ["AMPds2", "REFIT", "REDD", "UKDALE", "PRECON", "UCIPower", "Plegma"]

# Series longer than this are evenly subsampled (the length of one synthetic
# training case: one year at 15-min resolution).
TARGET_T = 35040
_SUBSAMPLE_CAP = TARGET_T

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
# Benchmark parsers (paper protocol)
# --------------------------------------------------------------------------- #
# UK-DALE metadata lists a second whole-house meter (the sound-card power meter,
# correlation 0.99 with the mains) as a child of the mains in buildings 1, 2, 5.
_UKDALE_DUPLICATE_MAINS = {"mains_second_phase", "main_house_meter_second_phase", "main_site_meter"}
_MIN_METERS = 3      # a case needs at least three non-constant meters
_PLEGMA_MIN_SUB = 3  # Plegma: at least three sub-meters (N >= 4)


def _even_index(n: int, target: int) -> np.ndarray:
    """``target`` evenly spaced row indices out of ``n`` (as in the paper)."""
    return np.linspace(0, n - 1, target).astype(np.int64)


def _read_numeric(csv_path: Path, max_rows: Optional[int], usecols=None) -> pd.DataFrame:
    """Read a CSV and coerce every non-time column to float (NaN on failure)."""
    df = pd.read_csv(csv_path, nrows=max_rows, usecols=usecols, low_memory=False)
    df.columns = [_clean_id(c) for c in df.columns]
    for c in df.columns:
        if not _is_time_column(c):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _hierarchy(dataset_dir: Path) -> dict:
    """The dataset's node map ``{meter_id: {"children": [...], ...}}``. A REFIT
    file without an explicit ``hierarchy`` gets the implicit tree of its labels."""
    payload = json.loads((dataset_dir / "hierarchy.json").read_text(encoding="utf-8"))
    hierarchy = payload.get("hierarchy")
    if isinstance(hierarchy, dict) and hierarchy:
        return hierarchy
    return {parent: {"children": kids} for parent, kids in _implicit_refit_edges(payload).items()}


def _children(node: dict) -> List[str]:
    return _children_ids(node) if isinstance(node, dict) else []


def _build_case(dataset: str, building: str, df: pd.DataFrame, cols: List[str],
                edges_named: List[Tuple[str, str]], min_meters: int = _MIN_METERS) -> Optional[dict]:
    """One benchmark case: the building's recording window, gaps filled forward
    then backward, at most :data:`TARGET_T` evenly spaced samples, constant
    meters dropped. ``None`` if fewer than ``min_meters`` meters remain."""
    if not cols:
        return None
    sub = df[cols].dropna(how="all")
    if sub.empty:
        return None
    data = sub.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
    if len(data) > TARGET_T:
        data = data[_even_index(len(data), TARGET_T)]
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    keep = data.var(axis=0) > 1e-12
    if keep.sum() < min_meters:
        return None
    names = [c for c, k in zip(cols, keep) if k]
    index = {n: i for i, n in enumerate(names)}
    true_edges = {(index[p], index[c]) for p, c in edges_named if p in index and c in index}
    return {"dataset": dataset, "building": building, "data": data[:, keep],
            "names": names, "true_edges": true_edges}


def _parse_ampds2(d: Path, max_rows: Optional[int]) -> List[dict]:
    df = _read_numeric(d / "consolidated.csv", max_rows)
    root = "WHE_P"
    cols = [root] + [c for c in df.columns if c.endswith("_P") and c != root]
    edges = [(root, c) for c in _children(_hierarchy(d).get(root))]
    case = _build_case("AMPds2", "house1", df, cols, edges)
    return [case] if case else []


def _parse_rooted(dataset: str, d: Path, max_rows: Optional[int], tag: str, root_suffix: str,
                  drop_labels: AbstractSet[str] = frozenset()) -> List[dict]:
    """Datasets with one root per building: ``<tag><k>_<root_suffix>``, all of the
    building's columns (sorted) as meters, the root's listed children as edges."""
    df = _read_numeric(d / "consolidated.csv", max_rows)
    h = _hierarchy(d)
    numbers = sorted({int(c[len(tag):].split("_", 1)[0]) for c in df.columns
                      if c.startswith(tag) and c[len(tag):].split("_", 1)[0].isdigit()})
    cases = []
    for k in numbers:
        prefix, root = f"{tag}{k}_", f"{tag}{k}_{root_suffix}"
        cols = sorted(c for c in df.columns if c.startswith(prefix))
        if root not in cols:
            continue
        kids = _children(h.get(root))
        drop = {c for c in kids if h.get(c, {}).get("label", "") in drop_labels}
        cols = [c for c in cols if c not in drop]
        case = _build_case(dataset, f"{tag}{k}", df, cols, [(root, c) for c in kids if c not in drop])
        if case:
            cases.append(case)
    return cases


def _parse_precon(d: Path, max_rows: Optional[int]) -> List[dict]:
    df = _read_numeric(d / "consolidated.csv", max_rows)
    h = _hierarchy(d)
    cases = []
    for root in [c for c in df.columns if c.endswith("_aggregate")]:
        kids = [c for c in _children(h.get(root)) if c in df.columns]
        if len(kids) < 3:
            continue
        case = _build_case("PRECON", root.split("_", 1)[0], df, [root] + kids, [(root, c) for c in kids])
        if case:
            cases.append(case)
    return cases


def _parse_uci(d: Path, max_rows: Optional[int]) -> List[dict]:
    root, kids = "Global_active_power", ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    df = _read_numeric(d / "consolidated.csv", max_rows, usecols=[root] + kids)
    df[root] = df[root] * 1000.0          # kW -> W
    df[kids] = df[kids] * 60.0            # Wh per minute -> W
    case = _build_case("UCIPower", "house1", df, [root] + kids, [(root, k) for k in kids])
    return [case] if case else []


def _parse_plegma(d: Path, max_rows: Optional[int]) -> List[dict]:
    df = _read_numeric(d / "consolidated.csv", max_rows)
    cases = []
    for root, node in _hierarchy(d).items():
        if node.get("type") != "aggregate":
            continue
        kids = [c for c in _children(node) if c in df.columns]
        if len(kids) < _PLEGMA_MIN_SUB:
            continue
        both = df[[root] + kids].dropna()
        if both.empty or float(both[kids].sum(axis=1).mean()) > float(both[root].mean()):
            continue  # sub-meters exceed the aggregate: not parent >= sum of children
        case = _build_case("Plegma", root.split("_", 1)[0], df, [root] + kids,
                           [(root, c) for c in kids], min_meters=_PLEGMA_MIN_SUB + 1)
        if case:
            cases.append(case)
    return cases


_PARSERS = {
    "AMPds2": _parse_ampds2,
    "REFIT": lambda d, m: _parse_rooted("REFIT", d, m, "house", "Aggregate"),
    "REDD": lambda d, m: _parse_rooted("REDD", d, m, "house", "main"),
    "UKDALE": lambda d, m: _parse_rooted("UKDALE", d, m, "bldg", "mains", _UKDALE_DUPLICATE_MAINS),
    "PRECON": _parse_precon,
    "UCIPower": _parse_uci,
    "Plegma": _parse_plegma,
}


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
            sub = sub.iloc[_even_index(len(sub), _SUBSAMPLE_CAP)]

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
        If given, only the first ``max_rows`` CSV rows are read. Forwarded to the
        dataset's benchmark parser, or to :func:`load_real_buildings` for
        datasets outside the benchmark.
    """
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; known: {sorted(DATASETS)}")
    dataset_dir = Path(root) / DATASETS[name]
    if name in _PARSERS:
        return _PARSERS[name](dataset_dir, max_rows)
    return load_real_buildings(dataset_dir, max_rows=max_rows)


def apply_apriori_filter(name: str, buildings: List[dict]) -> List[dict]:
    """Apply the paper's a-priori dataset rules.

    The benchmark parsers already apply the per-building rules (see the module
    docstring). RAE is excluded entirely: both houses have a computed mains (the
    exact sum of their circuits), so there is no physical whole-house root.
    """
    if name == "RAE":
        return []
    return buildings
