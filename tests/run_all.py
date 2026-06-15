#!/usr/bin/env python3
"""Run all tests without requiring pytest.

Usage:  python tests/run_all.py     (or: python -m pytest tests/)
"""
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import test_decode
import test_baselines
import test_smoke_avici
import test_smoke_gnn

CASES = [
    ("decode/recovers_chain", test_decode.test_recovers_chain),
    ("baselines/shapes", test_baselines.test_all_baselines_shapes),
    ("avici/smoke_cpu", test_smoke_avici.test_avici_smoke_cpu),
    ("gnn/train_save_reload", test_smoke_gnn.test_gnn_train_save_reload),
]


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
