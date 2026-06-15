# MeterHierarchyGNN

Reconstructing the **parent-child wiring hierarchy of energy (sub)meters** in
buildings from consumption time series alone, using a Graph Neural Network and a
suite of classical, optimization-based, and amortized baselines.

Every edge scorer (the GNN and each baseline) produces an `(N, N)` score matrix
where `S[i, j]` is the strength that meter `i` is the **parent** of meter `j`. A
shared **maximum-spanning-arborescence (Edmonds/Chu-Liu)** decoder turns that
matrix into a valid rooted tree, so all methods are compared on equal footing.

## Methods

| Family | Method | Module |
|---|---|---|
| Learned (ours) | Two-pass directed dual-embedding GNN (DEDGAT) | `meterhierarchy.model` |
| Amortized | **AVICI** (axial-attention transformer, trained on synthetic data) | `meterhierarchy.avici` |
| Classical | Chow-Liu MI, PC (partial corr.), Granger, Hidden-Load NNLS | `meterhierarchy.baselines` |
| Optimization | NOTEARS, DYNOTEARS | `meterhierarchy.baselines` |

## Installation

```bash
git clone <your-remote> meter-hierarchy-gnn
cd meter-hierarchy-gnn
python -m venv .venv && . .venv/Scripts/activate    # Windows; use bin/activate on Linux
pip install -e .[dev]
```

PyTorch must be installed with the wheel matching your backend (see
[`docs/amd_rocm.md`](docs/amd_rocm.md)):

```bash
# AMD / ROCm
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
# CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify the backend:

```bash
python -c "from meterhierarchy.utils import device_report; print(device_report())"
```

> **Note:** a plain `pip install torch` on Windows may pull the **CPU-only**
> build (`2.x+cpu`, `torch.cuda.is_available() == False`). The code still runs on
> CPU; install the ROCm wheel to use the AMD GPU.

## Quickstart

```bash
# 1. Generate a small synthetic dataset (code only; data is git-ignored)
python scripts/generate_data.py --n 200 --out data/synth --seed 42

# 2. Smoke-check everything quickly (no GPU needed)
python scripts/train_avici.py --smoke
python -m pytest tests/            # or: python tests/run_all.py

# 3. Train (hyperparameter defaults match configs/*.yaml)
python scripts/train_gnn.py   --data data/synth --out checkpoints/gnn_ensemble.pt
python scripts/train_avici.py --data data/synth --out checkpoints/avici.pt

# 4. Evaluate the GNN + all baselines (incl. AVICI) on a synthetic dataset
python scripts/run_baselines.py --data data/synth
python scripts/evaluate.py      --data data/synth --gnn checkpoints/gnn_ensemble.pt

# 5. Evaluate on the paper's REAL datasets (zero-shot transfer)
python scripts/evaluate_real.py --data-root "<path>/RealDataClean" \
    --datasets AMPds2,REFIT,REDD,UKDALE,RAE --baselines CL,AVICI --avici checkpoints/avici.pt --gnn checkpoints/gnn_ensemble.pt
```

The real datasets (`RealDataClean/<name>/{consolidated.csv,hierarchy.json}`) live
outside the repo and are loaded by `meterhierarchy.data.real_io`, which applies
the same a-priori rooted-tree filter as the paper (RAE: `house1` only; UK-DALE:
drop building 1).

## Repository layout

```
src/meterhierarchy/
  model/        GNN (gnn.py) + Edmonds/Chu-Liu decoder (decode.py)
  features/     58 pairwise + 15 node features + Laplacian positional encodings
  data/         synthetic case generator + case I/O (parquet + hierarchy.json)
  baselines/    score(data, names) -> (N, N) for CL, PC, Granger, HL, NOTEARS, DYNOTEARS, AVICI
  avici/        the AVICI model, its data adapter, and training loop
  utils/        device selection + metrics
scripts/        thin CLI wrappers
configs/        YAML hyperparameters (GNN ensemble, AVICI)
tests/          smoke + unit tests (runnable with or without pytest)
docs/           architecture, baselines, data format, AMD/ROCm notes
```

## Documentation

* [`docs/architecture.md`](docs/architecture.md) - how the GNN works, step by step
* [`docs/baselines.md`](docs/baselines.md) - the 7 baselines and the `score()` contract
* [`docs/data_format.md`](docs/data_format.md) - the on-disk case format
* [`docs/amd_rocm.md`](docs/amd_rocm.md) - running on AMD GPUs (ROCm)

## Data & results

The repository contains **code only**. Generated datasets, trained weights,
result tables, logs, and figures are intentionally git-ignored (see
[`.gitignore`](.gitignore)) and must be produced locally via the scripts above.

## License

MIT - see [`LICENSE`](LICENSE).
