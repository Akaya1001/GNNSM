"""GNN trains a tiny model, saves an ensemble checkpoint, reloads and predicts."""
import pathlib
import tempfile

import torch

from meterhierarchy.avici.train import make_smoke_cases
from meterhierarchy.data.case_io import build_case_features
from meterhierarchy.train_gnn import (
    compute_norm,
    apply_norm,
    train_one_model,
    ensemble_predict,
    save_ensemble,
    PE_DIM,
)
from meterhierarchy.evaluate import load_gnn_ensemble


def test_gnn_train_save_reload():
    device = torch.device("cpu")
    raws = make_smoke_cases(k=6, N=6, T=128, seed=0)
    cases = [
        build_case_features({"data": r["data"], "true_edges": r["true_edges"], "meta": {}})
        for r in raws
    ]
    splits = {"train": cases[:4], "val": cases[4:]}
    norm = compute_norm(splits["train"])
    apply_norm(splits["train"], norm)
    apply_norm(splits["val"], norm)

    cfg = {"name": "S", "hidden": 32, "layers": 3, "heads": 4, "dropout": 0.1, "lr": 2e-3}
    model, f1 = train_one_model("S", cfg, splits, PE_DIM, device, lambda *_a: None, n_epochs=3, patience=3)
    assert 0.0 <= f1 <= 1.0, f1

    out = pathlib.Path(tempfile.mkdtemp()) / "ens.pt"
    save_ensemble(out, [model], [cfg], norm, PE_DIM)
    models, norm2, pe = load_gnn_ensemble(out, device)
    assert len(models) == 1 and pe == PE_DIM
    probs = ensemble_predict(models, splits["val"][0], device)
    assert probs.shape == (splits["val"][0]["n_meters"],) * 2


if __name__ == "__main__":
    test_gnn_train_save_reload()
    print("test_smoke_gnn OK")
