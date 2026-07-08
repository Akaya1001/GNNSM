"""Sample hierarchical meter trees with constraints:
- n_meters in [4, 150]
- max_depth (levels, incl. leaves) in [2, 6]  -> depth labels 0..L-1 with L in [2,6]
- n_roots in [1, 5]
"""
from __future__ import annotations

import numpy as np


def _build_single_tree(rng: np.random.Generator, start_id: int, n_levels: int,
                       branch_min: int = 2, branch_max: int = 4):
    """Build one tree with n_levels levels (depth labels 0..n_levels-1).
    Returns list of node dicts with local ids (to be rewritten later) and
    the root local id."""
    nodes = []
    # BFS creation
    root_local = start_id
    nodes.append({
        "local_id": root_local,
        "parent_local_id": None,
        "depth": 0,
        "children_local_ids": [],
    })
    next_id = start_id + 1
    frontier = [root_local]
    for level in range(1, n_levels):
        new_frontier = []
        for parent_lid in frontier:
            n_children = int(rng.integers(branch_min, branch_max + 1))
            for _ in range(n_children):
                nodes.append({
                    "local_id": next_id,
                    "parent_local_id": parent_lid,
                    "depth": level,
                    "children_local_ids": [],
                })
                # append to parent's children list
                for n in nodes:
                    if n["local_id"] == parent_lid:
                        n["children_local_ids"].append(next_id)
                        break
                new_frontier.append(next_id)
                next_id += 1
        frontier = new_frontier
    return nodes, root_local, next_id


def sample_hierarchy(rng: np.random.Generator,
                     n_meters_range=(4, 150),
                     n_levels_range=(2, 6),
                     n_roots_range=(1, 5),
                     max_attempts: int = 200):
    """Sample a forest. Returns dict with 'meters' (list of dicts in reference
    JSON schema), and summary values.

    Each meter dict has: id, parent_id, children, depth, root_id, type,
    has_hidden_load, hidden_load_fraction (set later by aggregate module).
    """
    n_min, n_max = n_meters_range
    L_min, L_max = n_levels_range
    R_min, R_max = n_roots_range

    for _ in range(max_attempts):
        n_roots = int(rng.integers(R_min, R_max + 1))
        trees = []
        next_id = 0
        total = 0
        overall_max_levels = 0
        for r in range(n_roots):
            n_levels = int(rng.integers(L_min, L_max + 1))
            # Adjust branching to avoid exploding count for deep trees
            if n_levels >= 4:
                b_min, b_max = 2, 3
            else:
                b_min, b_max = 2, 4
            nodes, _root_lid, next_id = _build_single_tree(
                rng, next_id, n_levels, b_min, b_max
            )
            trees.append(nodes)
            total += len(nodes)
            overall_max_levels = max(overall_max_levels, n_levels)
            if total > n_max:
                break
        if n_min <= total <= n_max:
            break
    else:
        raise RuntimeError("Could not sample a valid hierarchy within constraints")

    # Flatten and re-id sequentially, build reference-schema dicts
    all_nodes = [node for t in trees for node in t]
    # map local -> meter_XXX string id
    id_map = {n["local_id"]: f"meter_{i:03d}" for i, n in enumerate(all_nodes)}
    # compute root_id per tree
    root_ids_per_tree = []
    for t in trees:
        root_local = t[0]["local_id"]
        root_ids_per_tree.append(id_map[root_local])
    # build final meters list in order of all_nodes
    meters = []
    # for lookup of tree index per local_id
    local_to_tree = {}
    for ti, t in enumerate(trees):
        for n in t:
            local_to_tree[n["local_id"]] = ti
    for n in all_nodes:
        mid = id_map[n["local_id"]]
        parent_id = id_map[n["parent_local_id"]] if n["parent_local_id"] is not None else None
        children = [id_map[c] for c in n["children_local_ids"]]
        root_id = root_ids_per_tree[local_to_tree[n["local_id"]]]
        if parent_id is None:
            node_type = "root"
        elif not children:
            node_type = "leaf"
        else:
            node_type = "intermediate"
        meters.append({
            "id": mid,
            "parent_id": parent_id,
            "children": children,
            "depth": n["depth"],
            "root_id": root_id,
            "type": node_type,
            "has_hidden_load": False,  # set later
            "hidden_load_fraction": 0.0,  # set later
        })

    # max_depth in reference = max depth label (0-indexed), so n_levels-1
    max_depth = max(m["depth"] for m in meters)
    return {
        "meters": meters,
        "n_meters": len(meters),
        "n_roots": len(trees),
        "max_depth": max_depth,
    }
