# How the GNN works

The model scores every ordered meter pair `(i, j)` as a candidate parent→child
edge, and a maximum-spanning-arborescence decoder turns those scores into a
rooted tree. Pipeline: **features → encode → directed attention → jumping
knowledge → heads → decode**.

## 1. Features (`features/pairwise.py`)
From a `(T, N)` consumption matrix:
* **58 edge features** `(N, N, 58)` per ordered pair: correlation (Pearson,
  diff, Spearman, partial), lag/aggregation correlation (directional), amplitude
  ratios, variance/covariance, overlap/co-activation, information-theoretic
  (MI/entropy). The *directional* (lagged/Granger) and *amplitude* features are
  what let the model tell parent from child — plain correlation is symmetric.
* **15 node features** `(N, 15)`: per-meter profile (magnitude, rank,
  variability, autocorrelation, entropy, ...).
* **8 Laplacian positional encodings** `(N, 8)`: spectral "address" of each node
  in the correlation graph.

## 2. Encoder (`model/gnn.py`)
Two MLPs (Linear + LayerNorm + GELU) lift node `[15 + 8 PE]` and edge `[58]`
features into a common `hidden_dim` (default 128).

## 3. DEDGAT — directed dual-embedding attention (the core)
Each of `L` layers runs **two directed attention streams** over the full `N×N`
graph: every node has *separate parent and child* query/key/value projections,
so the same node is represented differently as a potential parent vs. child.
Edge embeddings bias the attention scores; the two streams are gated, fused,
and passed through a transformer-style residual + feed-forward block. This is
what makes attention *directed* (unlike standard GAT).

## 4. Edge updater
After each attention layer, every edge `(i, j)` is refined from its endpoints
plus sibling context (row mean) and parent-competition context (column mean), so
edges are scored relative to their neighborhood (one parent per child; many
children per parent).

## 5. Jumping knowledge
Outputs of all layers are concatenated and projected, so the classifier sees
shallow (local) and deep (global) representations — counters over-smoothing.

## 6. Heads
Three heads on the final embeddings: an **edge classifier** `(N, N)` (the
parent→child logits), a **root** head (no-parent), and a **latent** head
(non-additive parent with hidden load). The latter two are auxiliary tasks.

## 7. NOTEARS acyclicity
`h(W) = tr(exp(W∘W)) − N` (4th-order Taylor) on the sigmoid edge scores is added
as a differentiable regularizer with linear warmup.

## 8. Two-pass wrapper (`TwoPassGNN`)
A small first pass produces preliminary edge logits; **6 latent edge features**
(raw probability, asymmetry, cycle indicator, child/parent competition, row
z-score) are derived from them and concatenated with the 58 features for a
deeper second pass that emits the final logits.

## 9. Training (`train_gnn.py`)
Loss = masked BCE on edge logits (dynamic `pos_weight` for the `O(N)` vs `O(N²)`
imbalance) + auxiliary pass-1 BCE + root BCE + latent BCE + acyclicity. Non-
additive cases up-weighted; AdamW + warmup/cosine; gradient accumulation. An
**A/C ensemble** is trained on synthetic data; edge/node/PE features are
z-normalized with train statistics stored in the checkpoint.

## 10. Decoding (`model/decode.py`)
The averaged edge-probability matrix is fed to **Edmonds/Chu-Liu** with a virtual
super-root (forces exactly one real root), yielding a valid rooted tree
`{child: parent}` that is scored against the ground truth with edge-level F1.
