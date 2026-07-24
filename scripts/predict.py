#!/usr/bin/env python3
"""Predict a meter hierarchy from CSV(s) with a trained GNN (see RUN_PRETRAINED.md)."""
import _bootstrap  # noqa: F401
from meterhierarchy.predict import main

if __name__ == "__main__":
    main()
