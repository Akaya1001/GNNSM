"""Unit tests for the config-driven pipeline (no training, fast)."""
import tempfile
from pathlib import Path

from meterhierarchy import pipeline


def _cfg(text: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pipeline.toml"
        p.write_text(text, encoding="utf-8")
        return pipeline.load_config(p)


def test_stages_canonical_order_and_validation():
    cfg = _cfg('[run]\nstages = ["evaluate", "generate", "train"]\n')
    assert pipeline.resolve_stages(cfg) == ["generate", "train", "evaluate"]
    try:
        pipeline.resolve_stages(_cfg('[run]\nstages = ["deploy"]\n'))
        raise AssertionError("unknown stage must raise")
    except SystemExit:
        pass


def test_methods_expansion_and_validation():
    from meterhierarchy.baselines import available

    cfg = _cfg('[run]\nmethods = ["all"]\n')
    methods = pipeline.resolve_methods(cfg)
    assert methods[0] == "GNN" and set(methods) == {"GNN", *available()}
    # de-dupe, order kept
    cfg = _cfg('[run]\nmethods = ["CL", "GNN", "CL"]\n')
    assert pipeline.resolve_methods(cfg) == ["CL", "GNN"]
    try:
        pipeline.resolve_methods(_cfg('[run]\nmethods = ["XGBoost"]\n'))
        raise AssertionError("unknown method must raise")
    except SystemExit:
        pass


def test_checkpoint_paths_follow_train_out():
    cfg = _cfg('[train.gnn]\nout = "ck/g.pt"\n[train.avici]\nout = "ck/a.pt"\n')
    assert pipeline._ckpt(cfg, "GNN") == Path("ck/g.pt")
    assert pipeline._ckpt(cfg, "AVICI") == Path("ck/a.pt")
    assert pipeline._ckpt({}, "GNN") == Path("checkpoints/gnn_ensemble.pt")


def test_missing_data_path_raises():
    try:
        pipeline.stage_generate(_cfg('[generate]\nn = 2\n'))
        raise AssertionError("missing [data].path must raise")
    except SystemExit:
        pass


if __name__ == "__main__":
    test_stages_canonical_order_and_validation()
    test_methods_expansion_and_validation()
    test_checkpoint_paths_follow_train_out()
    test_missing_data_path_raises()
    print("OK")
