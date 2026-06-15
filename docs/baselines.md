# Baselines and the scorer contract

Every edge scorer — the GNN and all baselines — implements the same contract, so
they share the Edmonds decoder and the F1 metric and are compared identically.

## The `score` contract

```python
def score(data: np.ndarray, names=None) -> np.ndarray:
    """data: (T, N) consumption matrix -> (N, N) float matrix S.
    S[i, j] = strength that meter i is the PARENT of meter j (higher = more likely).
    """
```

The decoder then runs `find_best_tree(N, S)` (maximum spanning arborescence) to
produce a valid rooted tree. Scorers are registered in
`meterhierarchy.baselines.REGISTRY` and fetched with `get_scorer(name)`.

## The methods

| Name | Module | Idea | S symmetry | Deps |
|---|---|---|---|---|
| `CL` | `chow_liu.py` | Chow-Liu: pairwise mutual information (max spanning tree) | symmetric | numpy |
| `PC` | `pc.py` | order-1 partial correlation (PC-style) | symmetric | numpy |
| `Granger` | `granger.py` | pairwise Granger F-test (lagged predictability) | directed | numpy |
| `HL` | `hidden_load.py` | hidden-load non-negative least squares reconstruction | directed | numpy, scipy |
| `NOTEARS` | `notears.py` | continuous DAG learning, `\|W_ij\|` | directed | numpy, scipy |
| `DYNOTEARS` | `dynotears.py` | dynamic (time-series) NOTEARS | directed | numpy, scipy |
| `AVICI` | `avici.py` | amortized transformer, trained on synthetic data (see below) | directed | torch |

Symmetric scorers (CL, PC) rely on the amplitude/decoder step to orient edges;
the directed scorers produce an asymmetric `S` directly.

## AVICI (the learned baseline)

AVICI is the conceptual peer of the GNN: an axial-attention transformer
(`meterhierarchy.avici.model`) trained **once** on the synthetic corpus
(`scripts/train_avici.py`), then applied zero-shot. Its `score()` loads the
checkpoint (`MH_AVICI_CKPT` env var or `--avici <path>`; default
`checkpoints/avici.pt`) and returns the `(N, N)` edge-probability matrix
`theta[i, j] = P(i is parent of j)`. If no checkpoint exists it returns zeros
with a warning, so the harness never crashes. See
[`architecture.md`](architecture.md) for the model details.

## Running

```bash
python scripts/run_baselines.py --data data/synth --baselines all
python scripts/run_baselines.py --data data/synth --baselines CL,PC,AVICI --avici checkpoints/avici.pt
```
