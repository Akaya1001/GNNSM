"""Case writer - produces hierarchy.json + meter.parquet matching the schema
expected by ``meterhierarchy.data.case_io.load_case``.

meter.parquet : a "timestamp" column plus one float64 column per meter id.
hierarchy.json: a dict with a "meters" list (each entry has "id" and
                "parent_id"; the root's parent_id is null), plus "n_roots",
                "max_depth", and "approx_sum_children_equals_parent".
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


FORMAT_NOTES = {
    "meter_csv_first_column": "timestamp",
    "meter_csv_other_columns": "one column per meter id",
    "meter_values": "non-cumulative interval values in kWh per 15 minutes",
}


def write_case(case_dir: Path, df: pd.DataFrame, index: pd.DatetimeIndex,
               meters: list, case_id: str, split: str,
               hidden_load_mode: str, consistency_metric: float,
               time_resolution_minutes: int = 15) -> None:
    """Write hierarchy.json and meter.parquet for a single case.

    df: values-only DataFrame with columns meter_XXX (no timestamp col yet).
    index: datetime index.
    meters: list of meter dicts (reference schema).
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    # --- meter.parquet: timestamp as a regular column, RangeIndex ---
    out = df.reset_index(drop=True).copy()
    out.insert(0, "timestamp", index)
    # ensure float64 for meter columns
    for col in out.columns:
        if col != "timestamp":
            out[col] = out[col].astype("float64")
    out.to_parquet(case_dir / "meter.parquet", index=False)

    # --- hierarchy.json ---
    hierarchy = {
        "case_id": case_id,
        "split": split,
        "time_resolution_minutes": time_resolution_minutes,
        "n_timestamps": int(len(index)),
        "n_meters": len(meters),
        "n_roots": sum(1 for m in meters if m["type"] == "root"),
        "max_depth": max(m["depth"] for m in meters),
        "approx_sum_children_equals_parent": False,
        "hidden_load_mode": hidden_load_mode,
        "consistency_metric_mean_abs_error_over_child_sum": float(consistency_metric),
        "format_notes": FORMAT_NOTES,
        "meters": meters,
    }
    with open(case_dir / "hierarchy.json", "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)
