"""Evaluate the GNN ensemble and/or the baselines on a dataset.

All methods share the Edmonds decoder and the edge-level F1 metric, so the
numbers are directly comparable. Baselines consume the raw ``(T, N)`` matrix;
the GNN consumes normalized features (using the statistics saved in its
checkpoint).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .model.gnn import TwoPassGNN
from .model.decode import find_best_tree
from .utils.device import get_device, device_report
from .utils.metrics import evaluate_f1
from .data.case_io import load_case, build_case_features, iter_case_dirs
from .train_gnn import apply_norm, ensemble_predict
from .baselines import get_scorer, available


def load_gnn_ensemble(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    norm = {k: np.asarray(v) for k, v in ckpt["norm"].items()}
    pe_dim = int(ckpt.get("pe_dim", 8))
    models = []
    for cfg, sd in zip(ckpt["ensemble_configs"], ckpt["ensemble_state_dicts"]):
        m = TwoPassGNN(
            n_edge_features=58, n_node_features=15, n_pe_features=pe_dim,
            hidden_dim=cfg["hidden"], n_layers=cfg["layers"],
            n_heads=cfg["heads"], dropout=cfg["dropout"],
        ).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models, norm, pe_dim


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def evaluate(
    data_dir: Path,
    split: str,
    gnn_ckpt: Optional[Path],
    baseline_names: List[str],
    device: torch.device,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    folders = iter_case_dirs(Path(data_dir) / split, limit=limit)
    raws = [r for r in (load_case(f) for f in folders) if r is not None and r["n_meters"] >= 2]
    if not raws:
        raise SystemExit(f"no cases under {data_dir}/{split}")

    results: Dict[str, List[float]] = {}

    if gnn_ckpt is not None:
        models, norm, pe_dim = load_gnn_ensemble(Path(gnn_ckpt), device)
        f1s = []
        for raw in raws:
            feat = build_case_features(raw, pe_dim=pe_dim)
            apply_norm([feat], norm)
            probs = ensemble_predict(models, feat, device)
            f1, _, _ = evaluate_f1(find_best_tree(raw["n_meters"], probs), raw["true_edges"])
            f1s.append(f1)
        results["GNN"] = f1s

    for name in baseline_names:
        scorer = get_scorer(name)
        f1s = []
        for raw in raws:
            try:
                S = scorer(raw["data"])
                f1, _, _ = evaluate_f1(find_best_tree(raw["n_meters"], S), raw["true_edges"])
            except Exception:
                f1 = 0.0
            f1s.append(f1)
        results[name] = f1s

    print(f"\nEvaluation on {data_dir}/{split}  ({len(raws)} cases)")
    print(f"{'Method':<12}{'F1 mean':>10}{'F1 median':>12}")
    print("-" * 34)
    summary = {}
    for name, f1s in results.items():
        summary[name] = _mean(f1s)
        print(f"{name:<12}{_mean(f1s):>10.4f}{float(np.median(f1s)) if f1s else 0.0:>12.4f}")
    return summary


def _parse_baselines(arg: str) -> List[str]:
    if arg.lower() == "all":
        return available()
    return [b.strip() for b in arg.split(",") if b.strip()]


def _pick_split(data_dir: Path, requested: str) -> str:
    if (Path(data_dir) / requested).exists():
        return requested
    for s in ("test", "val", "train"):
        if (Path(data_dir) / s).exists():
            return s
    return requested


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate the GNN ensemble and baselines.")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--gnn", type=str, default=None, help="GNN ensemble checkpoint")
    p.add_argument("--baselines", type=str, default="", help="comma list or 'all' (default: none)")
    p.add_argument("--avici", type=str, default=None, help="AVICI checkpoint path (sets MH_AVICI_CKPT)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "dml", "cpu"])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    if args.avici:
        os.environ["MH_AVICI_CKPT"] = args.avici
    device = get_device(args.device)
    print(device_report())
    split = _pick_split(Path(args.data), args.split)
    names = _parse_baselines(args.baselines) if args.baselines else []
    evaluate(Path(args.data), split, Path(args.gnn) if args.gnn else None, names, device, args.limit)


def main_baselines(argv: Optional[List[str]] = None) -> None:
    """Entry point for ``mh-run-baselines``: evaluate all baselines (no GNN)."""
    p = argparse.ArgumentParser(description="Evaluate all baselines on a dataset.")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--baselines", type=str, default="all")
    p.add_argument("--avici", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "dml", "cpu"])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    if args.avici:
        os.environ["MH_AVICI_CKPT"] = args.avici
    device = get_device(args.device)
    print(device_report())
    split = _pick_split(Path(args.data), args.split)
    evaluate(Path(args.data), split, None, _parse_baselines(args.baselines), device, args.limit)


def evaluate_real(
    data_root: Path,
    dataset_names: List[str],
    gnn_ckpt: Optional[Path],
    baseline_names: List[str],
    device: torch.device,
    max_rows: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate methods on the paper's real datasets (mean F1 over buildings).

    Applies the a-priori rooted-tree filter (RAE house1 only; UK-DALE drops
    building 1) before scoring.
    """
    from .data.real_io import load_real_dataset, apply_apriori_filter

    models = norm = None
    pe_dim = 8
    if gnn_ckpt is not None:
        models, norm, pe_dim = load_gnn_ensemble(Path(gnn_ckpt), device)

    methods = (["GNN"] if models else []) + list(baseline_names)
    table: Dict[str, Dict[str, float]] = {m: {} for m in methods}

    for name in dataset_names:
        try:
            buildings = apply_apriori_filter(name, load_real_dataset(Path(data_root), name, max_rows=max_rows))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip {name}] {exc}")
            continue
        for m in methods:
            f1s = []
            for b in buildings:
                N = b["data"].shape[1]
                try:
                    if m == "GNN":
                        feat = build_case_features(
                            {"data": b["data"], "true_edges": b["true_edges"], "meta": {}, "case_id": b["building"]},
                            pe_dim=pe_dim,
                        )
                        apply_norm([feat], norm)
                        S = ensemble_predict(models, feat, device)
                    else:
                        S = get_scorer(m)(b["data"])
                    f1, _, _ = evaluate_f1(find_best_tree(N, S), b["true_edges"])
                except Exception:  # noqa: BLE001
                    f1 = 0.0
                f1s.append(f1)
            table[m][name] = _mean(f1s)

    header = "Method".ljust(11) + "".join(f"{n:>9}" for n in dataset_names) + f"{'MEAN':>9}"
    print(f"\nReal-data evaluation (mean F1 over buildings; max_rows={max_rows})")
    print(header)
    print("-" * len(header))
    for m in methods:
        vals = [table[m].get(n, 0.0) for n in dataset_names]
        mean = sum(vals) / len(vals) if vals else 0.0
        print(m.ljust(11) + "".join(f"{v:>9.3f}" for v in vals) + f"{mean:>9.3f}")
    return table


_PALETTE = ["1f77b4", "ff7f0e", "2ca02c", "d62728", "9467bd", "8c564b", "e377c2", "17becf"]
_METHOD_ORDER = ["GNN", "CL", "PC", "Granger", "HL", "NOTEARS", "DYNOTEARS", "AVICI"]


def emit_latex(
    table: Dict[str, Dict[str, float]],
    dataset_names: List[str],
    out_dir: Path,
    gnn_ref: Optional[Dict[str, float]] = None,
    note: str = "",
) -> None:
    """Write a LaTeX table + a TikZ bar chart of the real-data results.

    Produces ``tab_real_eval_baselines.tex`` and ``fig_real_eval_baselines.tex``
    in ``out_dir``. ``gnn_ref`` optionally adds a GNN row from externally
    supplied per-dataset F1 (e.g. the paper's ensemble numbers).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: Dict[str, List[float]] = {}
    if gnn_ref:
        rows["GNN"] = [gnn_ref.get(d, float("nan")) for d in dataset_names]
    for m in table:
        rows[m] = [table[m].get(d, float("nan")) for d in dataset_names]
    methods = [m for m in _METHOD_ORDER if m in rows] + [m for m in rows if m not in _METHOD_ORDER]

    def _mean_row(vals: List[float]) -> float:
        fin = [v for v in vals if v == v]
        return sum(fin) / len(fin) if fin else float("nan")

    def _fmt(v: float) -> str:
        return "--" if v != v else f"{v:.3f}"

    ncol = len(dataset_names)
    T = [
        r"\begin{table}[pos=H]", r"\centering",
        r"\caption{Edge-level $F_1$ of the GNN and baselines on the real datasets "
        r"(mean over buildings; a-priori rooted-tree filter)." + (" " + note if note else "") + r"}",
        r"\label{tab:real-eval-baselines}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "r" * ncol + "r}",
        r"\toprule",
        "Method & " + " & ".join(dataset_names) + r" & MEAN \\",
        r"\midrule",
    ]
    for m in methods:
        vals = rows[m]
        cells = " & ".join(_fmt(v) for v in vals)
        T.append(f"{m} & {cells} & {_fmt(_mean_row(vals))} " + r"\\")
    T += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (out / "tab_real_eval_baselines.tex").write_text("\n".join(T), encoding="utf-8")

    sym = "{" + ",".join(dataset_names) + "}"
    F = ["% Auto-generated by meterhierarchy.evaluate.emit_latex (real-data baselines)"]
    for i, _m in enumerate(methods):
        F.append(f"\\providecolor{{mh{i}}}{{HTML}}{{{_PALETTE[i % len(_PALETTE)]}}}")
    F += [
        r"\begin{tikzpicture}", r"\begin{axis}[",
        r"  ybar=0.5pt, bar width=4pt, width=15cm, height=7cm, enlarge x limits=0.12,",
        f"  symbolic x coords={sym}, xtick=data,",
        r"  x tick label style={font=\small}, ymin=0, ymax=1.05, ylabel={$F_1$},",
        r"  grid=major, grid style={dashed, gray!30}, legend columns=-1,",
        r"  legend style={font=\footnotesize, at={(0.5,1.03)}, anchor=south},",
        r"]",
    ]
    for i, m in enumerate(methods):
        coords = " ".join(f"({d},{(v if v == v else 0.0):.3f})" for d, v in zip(dataset_names, rows[m]))
        F.append(f"  \\addplot[fill=mh{i}!75, draw=mh{i}] coordinates {{{coords}}};")
        F.append(f"  \\addlegendentry{{{m}}}")
    F += [r"\end{axis}", r"\end{tikzpicture}"]
    (out / "fig_real_eval_baselines.tex").write_text("\n".join(F), encoding="utf-8")
    print(f"wrote {out / 'tab_real_eval_baselines.tex'} and {out / 'fig_real_eval_baselines.tex'}")


def main_real(argv: Optional[List[str]] = None) -> None:
    """Entry point for ``mh-evaluate-real``."""
    p = argparse.ArgumentParser(description="Evaluate GNN/baselines on the paper's real datasets.")
    p.add_argument("--data-root", type=str, required=True, help="path to RealDataClean/")
    p.add_argument("--datasets", type=str, default="AMPds2,REFIT,REDD,UKDALE,RAE")
    p.add_argument("--gnn", type=str, default=None)
    p.add_argument("--baselines", type=str, default="", help="comma list or 'all'")
    p.add_argument("--avici", type=str, default=None, help="AVICI checkpoint (sets MH_AVICI_CKPT)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "dml", "cpu"])
    p.add_argument("--max-rows", type=int, default=4000, help="rows read per dataset (<0 = full, subsample 5000)")
    p.add_argument("--emit-latex", type=str, default=None, help="write LaTeX table + TikZ figure to this dir")
    p.add_argument("--gnn-ref", type=str, default=None, help="GNN per-dataset F1, e.g. 'AMPds2:0.855,REFIT:0.885'")
    p.add_argument("--note", type=str, default="", help="extra sentence appended to the emitted table caption")
    args = p.parse_args(argv)
    if args.avici:
        os.environ["MH_AVICI_CKPT"] = args.avici
    device = get_device(args.device)
    print(device_report())
    names = _parse_baselines(args.baselines) if args.baselines else []
    ds = [d.strip() for d in args.datasets.split(",") if d.strip()]
    mr = None if args.max_rows is not None and args.max_rows < 0 else args.max_rows
    table = evaluate_real(Path(args.data_root), ds, Path(args.gnn) if args.gnn else None, names, device, mr)
    if args.emit_latex:
        gnn_ref = None
        if args.gnn_ref:
            gnn_ref = {}
            for tok in args.gnn_ref.split(","):
                k, v = tok.split(":")
                gnn_ref[k.strip()] = float(v)
        emit_latex(table, ds, Path(args.emit_latex), gnn_ref=gnn_ref, note=args.note)


if __name__ == "__main__":
    main()
