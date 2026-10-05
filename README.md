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
git clone https://github.com/Akaya1001/GNNSM.git
cd GNNSM
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

# 3. Train (configs/*.yaml document the built-in defaults; they are not read by code)
python scripts/train_gnn.py   --data data/synth --out checkpoints/gnn_ensemble.pt
python scripts/train_avici.py --data data/synth --out checkpoints/avici.pt \
    --dim 96 --layers 6 --heads 6 --epochs 25   # the paper's AVICI setting (Table 13)

# 4. Evaluate the GNN + all baselines (incl. AVICI) on a synthetic dataset
python scripts/run_baselines.py --data data/synth
python scripts/evaluate.py      --data data/synth --gnn checkpoints/gnn_ensemble.pt

# 5. Evaluate on the paper's REAL datasets (zero-shot transfer)
python scripts/evaluate_real.py --data-root "<path>/RealDataClean" \
    --datasets AMPds2,REFIT,REDD,UKDALE,PRECON,UCIPower,Plegma --baselines all --avici checkpoints/avici.pt --gnn checkpoints/gnn_ensemble.pt
```

The real datasets live outside the repo in numbered folders, e.g.
`RealDataClean/07_AMPds2/{consolidated.csv,hierarchy.json}` (`06_REFIT`,
`19_REDD`, `18_UKDALE`, `08_PRECON`, `05_UCI_Power`, `10_Plegma`); the
short-name → folder mapping lives in `meterhierarchy.data.real_io.DATASETS`.
The benchmark parsers follow the paper's protocol (Sec. 5.1.2, Tables 11 and 13):

* one case per building, rooted at its physical whole-house meter with at least
  three sub-meters (N ≥ 4); constant sensors are dropped;
* UK-DALE: all five buildings; the duplicate whole-house (sound-card) meter that
  the metadata lists under the mains of buildings 1, 2 and 5 is removed;
* UCI Power: `Global_active_power` (kW → W) with `Sub_metering_1..3`
  (Wh/min → W); the computed `remainder` is not a meter;
* Plegma: houses with at least three sub-meters whose sum does not exceed the
  aggregate (houses 1, 3, 4, 7, 11);
* RAE is excluded: its mains is the computed sum of its circuits.

The GNN sees the full series (at most 35,040 samples); every baseline sees it
evenly subsampled to 2,000 samples. Pass `--max-rows 4000` for a quick check on
the first 4,000 rows only.

### One-command pipeline

Steps 1-5 can also be driven by a single TOML config that selects the stages
(`generate`, `train`, `evaluate`), the methods to compare, and the data source
(synthetic or real):

```bash
python scripts/pipeline.py --config configs/pipeline.toml
```

See [`configs/pipeline.toml`](configs/pipeline.toml) for the annotated
reference config (e.g. train on the synthetic corpus, then evaluate zero-shot
on the real datasets by setting `source = "real"`).

### Predict on your own data (inference only)

To apply a trained ensemble to a raw meter CSV (one time column, one column per
meter) and print the predicted parent-child hierarchy, no ground truth needed:

```bash
python scripts/predict.py --gnn checkpoints/gnn_ensemble.pt --csv my_meters.csv --out tree.json
```

## Repository layout

```
src/meterhierarchy/
  model/        GNN (gnn.py) + Edmonds/Chu-Liu decoder (decode.py)
  features/     58 pairwise + 15 node features + Laplacian positional encodings
                (pure NumPy; the paper's runs computed them with Numba kernels)
  data/         synthetic case generator + case I/O (parquet + hierarchy.json)
  baselines/    score(data, names) -> (N, N) for CL, PC, Granger, HL, NOTEARS, DYNOTEARS, AVICI
  avici/        the AVICI model, its data adapter, and training loop
  utils/        device selection + metrics
scripts/        thin CLI wrappers
configs/        pipeline.toml (read by scripts/pipeline.py) + reference YAML of the training defaults
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
