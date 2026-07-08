"""Config-driven pipeline: generate -> train -> evaluate in one command.

A single TOML config (see ``configs/pipeline.toml``) selects the data source
(synthetic corpus or the real datasets), the stages to run, and the methods to
compare. Methods are resolved through the baseline registry plus ``"GNN"`` and
share the Edmonds decoder and edge-level F1, so results are directly
comparable. Stages always execute in the canonical order
``generate -> train -> evaluate`` regardless of their order in the config.

Typical uses:
  * full synthetic study: ``stages = ["generate", "train", "evaluate"]``
  * zero-shot transfer: train on ``[data].path``, evaluate with
    ``source = "real"`` (the paper's setup)
  * quick comparison on existing data/checkpoints: ``stages = ["evaluate"]``
"""
from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

STAGES = ("generate", "train", "evaluate")


def load_config(path: Path) -> Dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def resolve_stages(cfg: Dict) -> List[str]:
    """Validate ``[run].stages`` and return them in canonical order."""
    requested = cfg.get("run", {}).get("stages", ["evaluate"])
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; known: {list(STAGES)}")
    return [s for s in STAGES if s in requested]


def resolve_methods(cfg: Dict) -> List[str]:
    """Expand ``[run].methods`` (``"all"`` = GNN + every baseline), validated."""
    from .baselines import available

    requested = cfg.get("run", {}).get("methods", ["all"])
    if "all" in requested:
        requested = ["GNN"] + available()
    known = {"GNN", *available()}
    unknown = [m for m in requested if m not in known]
    if unknown:
        raise SystemExit(f"unknown method(s) {unknown}; known: {sorted(known)}")
    return list(dict.fromkeys(requested))  # de-dupe, keep order


def _data_path(cfg: Dict, why: str) -> str:
    path = cfg.get("data", {}).get("path")
    if not path:
        raise SystemExit(f"[data].path is required for {why}")
    return str(path)


def _ckpt(cfg: Dict, method: str) -> Path:
    """Checkpoint path for a learned method; [train.<m>].out is the single source of truth."""
    default = f"checkpoints/{'gnn_ensemble' if method == 'GNN' else 'avici'}.pt"
    return Path(cfg.get("train", {}).get(method.lower(), {}).get("out", default))


def stage_generate(cfg: Dict) -> None:
    from .data.generator import main as generate_main

    g = cfg.get("generate", {})
    argv = ["--n", str(g.get("n", 200)), "--out", _data_path(cfg, "generate"),
            "--seed", str(g.get("seed", 42))]
    if "split_ratio" in g:
        argv += ["--split-ratio", str(g["split_ratio"])]
    generate_main(argv)


def stage_train(cfg: Dict, methods: List[str], device: str) -> None:
    """Train the learned methods among ``methods`` (classical ones need no training)."""
    data = _data_path(cfg, "train")
    if "GNN" in methods:
        from .train_gnn import main as gnn_main

        t = cfg.get("train", {}).get("gnn", {})
        argv = ["--data", data, "--out", str(_ckpt(cfg, "GNN")), "--device", device,
                "--epochs", str(t.get("epochs", 150)), "--patience", str(t.get("patience", 20))]
        gnn_main(argv)
    if "AVICI" in methods:
        from .avici.train import main as avici_main

        t = cfg.get("train", {}).get("avici", {})
        argv = ["--data", data, "--out", str(_ckpt(cfg, "AVICI")), "--device", device]
        for key, flag in (("epochs", "--epochs"), ("lr", "--lr"), ("dim", "--dim"),
                          ("layers", "--layers"), ("heads", "--heads"),
                          ("n_samples", "--n-samples"), ("max_meters", "--max-meters"),
                          ("cap_timesteps", "--cap-timesteps")):
            if key in t:
                argv += [flag, str(t[key])]
        avici_main(argv)


def stage_evaluate(cfg: Dict, methods: List[str], device) -> Dict:
    from .evaluate import evaluate, evaluate_real, emit_latex, _pick_split

    baselines = [m for m in methods if m != "GNN"]
    gnn_ckpt: Optional[Path] = None
    if "GNN" in methods:
        gnn_ckpt = _ckpt(cfg, "GNN")
        if not gnn_ckpt.exists():
            raise SystemExit(f"GNN checkpoint {gnn_ckpt} not found; run the train stage or fix [train.gnn].out")
    if "AVICI" in baselines:
        avici_ckpt = _ckpt(cfg, "AVICI")
        if avici_ckpt.exists():
            os.environ["MH_AVICI_CKPT"] = str(avici_ckpt)
        else:
            print(f"[warn] AVICI checkpoint {avici_ckpt} not found; AVICI scores fall back to zeros")

    source = cfg.get("data", {}).get("source", "synthetic")
    if source == "synthetic":
        data = Path(_data_path(cfg, "synthetic evaluation"))
        split = _pick_split(data, cfg["data"].get("split", "test"))
        return evaluate(data, split, gnn_ckpt, baselines, device, cfg["data"].get("limit"))
    if source == "real":
        r = cfg.get("data", {}).get("real", {})
        if not r.get("root"):
            raise SystemExit("[data.real].root is required when source = 'real'")
        datasets = list(r.get("datasets", ["AMPds2", "REFIT", "REDD", "UKDALE", "RAE"]))
        max_rows = r.get("max_rows", 4000)
        table = evaluate_real(Path(r["root"]), datasets, gnn_ckpt, baselines, device,
                              None if max_rows < 0 else max_rows)
        latex_dir = cfg.get("output", {}).get("latex_dir")
        if latex_dir:
            emit_latex(table, datasets, Path(latex_dir))
        return table
    raise SystemExit(f"unknown [data].source {source!r} (expected 'synthetic' or 'real')")


def run(config_path: Path) -> Optional[Dict]:
    from .utils.device import get_device, device_report

    cfg = load_config(config_path)
    stages = resolve_stages(cfg)
    methods = resolve_methods(cfg)
    device_flag = str(cfg.get("run", {}).get("device", "auto"))
    device = get_device(device_flag)

    print(device_report())
    print(f"pipeline: stages={stages} methods={methods} "
          f"source={cfg.get('data', {}).get('source', 'synthetic')}")

    results: Optional[Dict] = None
    for stage in stages:
        print(f"\n=== stage: {stage} ===")
        if stage == "generate":
            stage_generate(cfg)
        elif stage == "train":
            stage_train(cfg, methods, device_flag)
        else:
            results = stage_evaluate(cfg, methods, device)

    summary = cfg.get("output", {}).get("summary")
    if summary and results is not None:
        out = Path(summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"config": str(config_path), "stages": stages,
                                   "methods": methods, "results": results}, indent=2),
                       encoding="utf-8")
        print(f"wrote summary -> {out}")
    return results


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Config-driven generate/train/evaluate pipeline.")
    p.add_argument("--config", type=str, default="configs/pipeline.toml", help="TOML pipeline config")
    args = p.parse_args(argv)
    run(Path(args.config))


if __name__ == "__main__":
    main()
