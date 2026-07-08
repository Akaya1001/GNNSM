#!/usr/bin/env python3
"""Run the config-driven pipeline (see configs/pipeline.toml)."""
import _bootstrap  # noqa: F401
from meterhierarchy.pipeline import main

if __name__ == "__main__":
    main()
