"""Synthetic meter-hierarchy generation modules.

Sub-modules:

- :mod:`hierarchy`  -- sample random forests of meters with depth/branch/root
  constraints.
- :mod:`timeseries` -- per-leaf consumption series from device archetypes plus
  a per-root shared exogenous driver (temperature / activity / regime).
- :mod:`aggregate`  -- bottom-up aggregation with an additive hidden-load term.
- :mod:`anomalies`  -- the four anomaly types plus linear meter drift.
- :mod:`io_case`    -- writer for ``meter.parquet`` + ``hierarchy.json``.
"""
