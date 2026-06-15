#!/usr/bin/env python3
"""Evaluate all baselines (CL, PC, Granger, HL, NOTEARS, DYNOTEARS, AVICI) on a dataset."""
import _bootstrap  # noqa: F401
from meterhierarchy.evaluate import main_baselines

if __name__ == "__main__":
    main_baselines()
