"""Generate synthetic hierarchical meter cases.

Each case is one building/installation written to
``{out}/{split}/case_XXXX/`` as ``meter.parquet`` (a "timestamp" column plus
one float column per meter) and ``hierarchy.json`` (the ground-truth tree).
The output is readable by :func:`meterhierarchy.data.case_io.load_case`.

CLI::

    python -m meterhierarchy.data.generator --n 1000 --out ./data --seed 42 \
        --split-ratio 70,15,15

Splits default to 70/15/15 (train/val/test); ``--n`` sets the number of train
cases and val/test counts are derived from the ratio.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .genmodules.hierarchy import sample_hierarchy
from .genmodules.timeseries import (
    generate_leaf, build_time_index, build_shared_exogenous,
)
from .genmodules.aggregate import build_parent_series, classify_hidden_load_mode
from .genmodules.anomalies import apply_all_to_leaf
from .genmodules.io_case import write_case


N_TIMESTAMPS = 35040  # 1 year at 15-min
TIME_RES_MIN = 15
START_DATE = "2023-01-01"


def generate_case(case_idx: int, split: str, seed: int, output_root: Path) -> None:
    rng = np.random.default_rng(seed)

    # 1. Hierarchy
    hier = sample_hierarchy(rng)
    meters = hier["meters"]
    # Group leaves by root so we can give each root its own shared exogenous
    leaves_by_root: dict[str, list] = {}
    for m in meters:
        if not m["children"]:
            leaves_by_root.setdefault(m["root_id"], []).append(m["id"])

    index = build_time_index(N_TIMESTAMPS, start=START_DATE, freq=f"{TIME_RES_MIN}min")

    # 2. Generate shared exogenous per root + leaves coupled to it
    leaf_data = {}
    archetypes_used = []
    shared_by_root: dict[str, dict] = {}
    for root_id, leaf_ids in leaves_by_root.items():
        shared = build_shared_exogenous(N_TIMESTAMPS, rng)
        shared_by_root[root_id] = shared
        for lid in leaf_ids:
            x, arche = generate_leaf(N_TIMESTAMPS, rng, shared=shared)
            x = apply_all_to_leaf(x, index, rng)
            leaf_data[lid] = x
            archetypes_used.append(arche)

    leaves_df = pd.DataFrame(leaf_data, index=index)

    # 3. Aggregation with additive hidden load.
    # For aggregation, fill NaN with per-leaf MEAN (not zero) so parent
    # magnitudes stay realistic during leaf outages.
    leaves_for_agg = leaves_df.copy()
    col_means = leaves_for_agg.mean()
    leaves_for_agg = leaves_for_agg.fillna(col_means)
    # Use the same shared signal as used for leaves (pick first root's shared
    # for hidden load coupling - reasonable proxy since hidden load is
    # residential + HVAC which correlates with temperature).
    shared_any = next(iter(shared_by_root.values())) if shared_by_root else None
    all_df, mean_ratio, consistency = build_parent_series(
        leaves_for_agg, meters, rng, shared=shared_any,
    )
    # Restore NaNs on leaf columns (parents keep their aggregated values)
    for lid in leaves_df.columns:
        all_df[lid] = leaves_df[lid].values

    hidden_load_mode = classify_hidden_load_mode(mean_ratio)

    # 4. Write
    case_id = f"case_{case_idx:04d}"
    case_dir = output_root / split / case_id
    write_case(
        case_dir=case_dir,
        df=all_df,
        index=index,
        meters=meters,
        case_id=case_id,
        split=split,
        hidden_load_mode=hidden_load_mode,
        consistency_metric=consistency,
        time_resolution_minutes=TIME_RES_MIN,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate synthetic hierarchical meter cases."
    )
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of train cases (val/test derived from ratio)")
    parser.add_argument("--out", type=str, required=True,
                        help="Output root directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", dest="split_ratio", type=str,
                        default="70,15,15",
                        help="train,val,test split in percent (default 70,15,15)")
    args = parser.parse_args(argv)

    tr, va, te = (int(x) for x in args.split_ratio.split(","))
    assert tr + va + te == 100, "Split ratios must sum to 100"

    n_train = args.n
    n_val = round(n_train * va / tr)
    n_test = round(n_train * te / tr)

    output_root = Path(args.out)
    print(f"Generating {n_train} train / {n_val} val / {n_test} test cases "
          f"-> {output_root}")

    base_seed = args.seed
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw):
            return x

    # Per-split seed offsets keep the three splits on disjoint RNG streams.
    splits = [("train", n_train, 0), ("val", n_val, 1_000_000),
              ("test", n_test, 2_000_000)]
    for split_name, n_cases, offset in splits:
        print(f"\n[{split_name}] {n_cases} cases")
        for i in tqdm(range(n_cases), desc=split_name):
            generate_case(i, split_name, base_seed + offset + i, output_root)

    print(f"\nDone. Output at: {output_root}")


if __name__ == "__main__":
    main()
