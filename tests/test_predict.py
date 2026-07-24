"""Unit test for predict.load_meter_csv (CSV parsing only, no model)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from meterhierarchy.predict import load_meter_csv


def test_load_meter_csv_drops_time_and_constant():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.csv"
        pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=50, freq="15min"),
            "meter_a": rng.normal(size=50),
            "meter_b": rng.normal(size=50),
            "meter_const": np.ones(50),   # constant -> discarded
        }).to_csv(p, index=False)
        data, names = load_meter_csv(p)
        assert names == ["meter_a", "meter_b"], names   # time + constant dropped
        assert data.shape == (50, 2), data.shape


if __name__ == "__main__":
    test_load_meter_csv_drops_time_and_constant()
    print("OK")
