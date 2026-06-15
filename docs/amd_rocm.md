# Running on AMD GPUs (ROCm), and other backends

All compute device selection goes through `meterhierarchy.utils.device.get_device()`,
which resolves in this order: **CUDA/ROCm → DirectML → CPU**. The model code is
backend-agnostic; no `.cuda()` calls are hardcoded.

Check what you have:

```bash
python -c "from meterhierarchy.utils import device_report; print(device_report())"
```

## ROCm (recommended for AMD)

PyTorch's ROCm build exposes the **same `torch.cuda` API** as CUDA, so
`torch.cuda.is_available()` returns `True` and the code runs unchanged.

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
python -c "import torch; print(torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

RDNA4 cards (e.g. the Radeon AI PRO R9700, `gfx1201`) need a recent ROCm. If the
installed ROCm runtime does not yet advertise `gfx1201`, force a compatible
target before launching Python:

```bash
export HSA_OVERRIDE_GFX_VERSION=12.0.0   # Linux / WSL2
# PowerShell: $env:HSA_OVERRIDE_GFX_VERSION = "12.0.0"
```

Then validate with the smoke run (`python scripts/train_avici.py --smoke --device cuda`).

## CPU

A plain `pip install torch` on Windows commonly installs the **CPU-only** build
(`2.x+cpu`, `torch.cuda.is_available() == False`). Everything still runs, just
slower. The smoke runs and the test suite are CPU-friendly:

```bash
python scripts/train_avici.py --smoke --device cpu
python tests/run_all.py
```

## DirectML (Windows + AMD, `torch-directml`) — limited

`torch-directml` is auto-detected as a fallback, but it currently lacks several
operators this project relies on (advanced/boolean indexing, some attention
shapes). The AVICI loss was written with mask-multiplies to avoid boolean
indexing, but full training/inference is **not reliable on DirectML** today.
**Use ROCm or CPU.** If you only have DirectML, run on CPU (`--device cpu`).

## Notes

* No mixed precision (AMP) is used, so there is nothing backend-specific to
  configure there.
* Numba is **not** required (features are pure NumPy); installing it changes
  nothing.
