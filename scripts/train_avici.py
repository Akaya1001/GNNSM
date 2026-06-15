#!/usr/bin/env python3
"""Train AVICI on the synthetic corpus (use --smoke for a quick check)."""
import _bootstrap  # noqa: F401
from meterhierarchy.avici.train import main

if __name__ == "__main__":
    main()
