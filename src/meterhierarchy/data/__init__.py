"""Synthetic case generation and on-disk case I/O.

A *case* is one building/installation stored as ``meter.parquet`` (the (T, N)
consumption time series) plus ``hierarchy.json`` (metadata + ground-truth tree).
"""

from .case_io import load_case, build_case_features, iter_case_dirs

__all__ = ["load_case", "build_case_features", "iter_case_dirs"]
