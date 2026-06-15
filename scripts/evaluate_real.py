#!/usr/bin/env python3
"""Evaluate the GNN ensemble and/or baselines on the paper's REAL datasets.

Example:
  python scripts/evaluate_real.py --data-root "<...>/RealDataClean" \
      --datasets AMPds2,RAE,PRECON --baselines CL,AVICI --avici checkpoints/avici.pt
"""
import _bootstrap  # noqa: F401
from meterhierarchy.evaluate import main_real

if __name__ == "__main__":
    main_real()
