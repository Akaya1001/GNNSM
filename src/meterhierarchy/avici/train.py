"""Train AVICI on the synthetic meter-hierarchy corpus, or run a smoke check.

Loss: masked binary cross-entropy between the predicted edge logits and the
ground-truth adjacency, with a per-case positive-class weight to counter the
``O(N)`` edges vs ``O(N^2)`` non-edges imbalance. Trains once; the resulting
checkpoint is used zero-shot by :mod:`meterhierarchy.baselines.avici`.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.device import get_device, device_report
from .model import AVICIConfig, AVICIModel
from .data import case_to_input, adjacency_from_edges


def _edge_bce(logits: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Masked (off-diagonal) BCE with a per-case positive weight.

    Uses a float mask multiply (not boolean indexing) so it runs on backends
    with limited advanced-indexing support, e.g. DirectML.
    """
    d = A.shape[0]
    mask = 1.0 - torch.eye(d, device=A.device, dtype=A.dtype)  # off-diagonal selector
    n_off = float(mask.sum().item())
    n_pos = float((A * mask).sum().item())
    n_neg = max(n_off - n_pos, 1.0)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=A.device, dtype=logits.dtype)
    loss_elem = F.binary_cross_entropy_with_logits(
        logits, A, pos_weight=pos_weight, reduction="none"
    )
    return (loss_elem * mask).sum() / max(n_off, 1.0)


def train_avici(
    cases: List[dict],
    cfg: AVICIConfig,
    *,
    device: torch.device,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    n_samples: int = 400,
    warmup_steps: int = 50,
    grad_clip: float = 1.0,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> Dict:
    """Train an :class:`AVICIModel` in place; return ``{model, history}``.

    ``cases`` is a list of dicts with ``data`` ``(T, N)`` and ``true_edges``
    (a set of ``(parent, child)`` index tuples).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AVICIModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(step: int) -> float:
        return min(1.0, (step + 1) / max(1, warmup_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history: List[float] = []
    step = 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(cases))
        ep_loss, n = 0.0, 0
        for idx in order:
            case = cases[idx]
            data = case["data"]
            N = data.shape[1]
            if N < 2:
                continue
            inp = case_to_input(data, n_samples=n_samples, rng=rng).to(device)
            A = adjacency_from_edges(case["true_edges"], N).to(device)
            logits = model(inp)
            loss = _edge_bce(logits, A)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            step += 1
            ep_loss += float(loss.item())
            n += 1
        avg = ep_loss / max(n, 1)
        history.append(avg)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            log(f"  [AVICI] epoch {epoch + 1:>3d}/{epochs} | loss={avg:.4f} | lr={opt.param_groups[0]['lr']:.2e}")
    return {"model": model, "history": history}


def save_checkpoint(model: AVICIModel, path: Path, extra: Optional[Dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "config": model.cfg.to_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# --------------------------------------------------------------------------
# Smoke check (no real data, no GPU required)
# --------------------------------------------------------------------------
def _random_tree_edges(N: int, rng: np.random.Generator):
    """Random rooted tree: each non-root node gets one earlier-indexed parent."""
    edges = set()
    for child in range(1, N):
        parent = int(rng.integers(0, child))
        edges.add((parent, child))
    return edges


def make_smoke_cases(k: int = 4, N: int = 6, T: int = 128, seed: int = 0) -> List[dict]:
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(k):
        edges = _random_tree_edges(N, rng)
        # children loosely follow parents -> a learnable (non-trivial) signal
        x = rng.standard_normal((T, N)).astype(np.float32)
        for p, c in sorted(edges):
            x[:, c] += 0.6 * x[:, p]
        cases.append({"data": x, "true_edges": edges})
    return cases


def run_smoke(device: Optional[torch.device] = None, log: Callable[[str], None] = print) -> Dict:
    """Tiny end-to-end training run; asserts the loss is finite and decreases."""
    device = device or get_device()
    log(device_report())
    log(f"[smoke] device = {device}")
    cfg = AVICIConfig(dim=32, n_layers=2, n_heads=2, ffn_dim=64, dropout=0.0)
    cases = make_smoke_cases(k=4, N=6, T=128, seed=0)
    out = train_avici(
        cases, cfg, device=device, epochs=40, lr=1e-3, n_samples=64, warmup_steps=10, log=log
    )
    hist = out["history"]
    first, last = hist[0], hist[-1]
    log(f"[smoke] loss {first:.4f} -> {last:.4f}")
    assert np.isfinite(first) and np.isfinite(last), "loss is not finite"
    assert last < first, f"loss did not decrease ({first:.4f} -> {last:.4f})"
    # output-shape sanity
    inp = case_to_input(cases[0]["data"], n_samples=64).to(device)
    probs = out["model"].predict_matrix(inp).cpu().numpy()
    assert probs.shape == (6, 6), f"bad output shape {probs.shape}"
    log("[smoke] OK")
    return out


def _load_cases_from_dir(
    data_dir: Path,
    split: str,
    limit: Optional[int],
    max_meters: Optional[int] = None,
    cap_timesteps: Optional[int] = None,
) -> List[dict]:
    from ..data.case_io import load_case, iter_case_dirs

    folders = iter_case_dirs(Path(data_dir) / split, limit=limit)
    cases = []
    for folder in folders:
        raw = load_case(folder)
        if raw is None or raw["n_meters"] < 2:
            continue
        if max_meters is not None and raw["n_meters"] > max_meters:
            continue  # keep small-N cases (useful for fast CPU training)
        data = raw["data"]
        # Cap the per-case series length at load to bound RAM (the full year-long
        # series would otherwise be held for every case across all epochs).
        if cap_timesteps is not None and data.shape[0] > cap_timesteps:
            idx = np.unique(np.linspace(0, data.shape[0] - 1, cap_timesteps).round().astype(int))
            data = data[idx]
        cases.append({"data": data.astype(np.float32), "true_edges": raw["true_edges"]})
    return cases


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Train AVICI on the synthetic meter corpus.")
    p.add_argument("--data", type=str, default=None, help="dataset root containing a 'train' split")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--limit", type=int, default=None, help="cap number of scanned case folders")
    p.add_argument("--max-meters", type=int, default=None, help="skip cases with more than this many meters (faster on CPU)")
    p.add_argument("--cap-timesteps", type=int, default=None, help="subsample each case to at most this many timesteps at load (saves RAM)")
    p.add_argument("--out", type=str, default="checkpoints/avici.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "dml", "cpu"])
    p.add_argument("--smoke", action="store_true", help="run a tiny smoke check and exit")
    args = p.parse_args(argv)

    device = get_device(args.device)
    if args.smoke:
        run_smoke(device=device)
        return

    if not args.data:
        raise SystemExit("--data is required (or use --smoke). Generate one with scripts/generate_data.py")
    print(device_report())
    cases = _load_cases_from_dir(Path(args.data), args.split, args.limit, args.max_meters, args.cap_timesteps)
    print(f"loaded {len(cases)} training cases from {args.data}/{args.split}")
    if not cases:
        raise SystemExit("no cases found")
    cfg = AVICIConfig(dim=args.dim, n_layers=args.layers, n_heads=args.heads)
    t0 = time.perf_counter()
    out = train_avici(cases, cfg, device=device, epochs=args.epochs, lr=args.lr, n_samples=args.n_samples)
    save_checkpoint(out["model"], Path(args.out), extra={"n_samples": args.n_samples})
    print(f"saved checkpoint -> {args.out}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
