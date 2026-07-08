"""Backend-agnostic device selection.

PyTorch's ROCm build (AMD GPUs) exposes the *same* ``torch.cuda`` API as the
CUDA build, so ``torch.cuda.is_available()`` returns ``True`` on a working ROCm
install and no special handling is required. ``"auto"`` resolves to CUDA/ROCm
or CPU only; DirectML must be requested explicitly (``"dml"``) because it lacks
ops these models need (e.g. ``aten::eye`` on-device) and would crash mid-run.
"""
from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Return the best available compute device.

    Args:
        prefer: ``"auto"`` (default), ``"cuda"`` (CUDA or ROCm), ``"dml"``
            (DirectML / Windows AMD; explicit opt-in only) or ``"cpu"``.

    Resolution order for ``"auto"``: CUDA/ROCm -> CPU.
    """
    prefer = prefer.lower()
    if prefer == "cpu":
        return torch.device("cpu")

    # CUDA and ROCm both surface through torch.cuda.
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")

    # DirectML only on explicit request: its op coverage is too incomplete for
    # these models, so "auto" must never silently select it.
    if prefer == "dml":
        try:
            import torch_directml  # type: ignore

            return torch_directml.device()
        except Exception:
            pass

    return torch.device("cpu")


def device_report() -> str:
    """Human-readable one-line summary of the active backend."""
    if torch.cuda.is_available():
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown"
        ver = getattr(torch.version, "hip", None) or getattr(torch.version, "cuda", None)
        return f"{backend} GPU available: {name} (torch {torch.__version__}, backend {ver})"
    try:
        import torch_directml  # type: ignore

        if torch_directml.is_available():
            return f"DirectML device available (torch {torch.__version__})"
    except Exception:
        pass
    return f"No GPU backend detected; using CPU (torch {torch.__version__})"
