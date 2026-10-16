import numpy as np


def _assign_by_class(y, num_nodes, proportions, rng):
    """proportions: (num_classes, num_nodes) rows sum to 1."""
    indices = [[] for _ in range(num_nodes)]
    y = np.asarray(y)
    classes = np.unique(y)
    for cls_id, cls in enumerate(classes):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        cuts = (np.cumsum(proportions[cls_id]) * len(cls_idx)).astype(int)
        cuts = np.clip(cuts, 0, len(cls_idx))
        start = 0
        for node_id, end in enumerate(cuts):
            if node_id == num_nodes - 1:
                end = len(cls_idx)
            indices[node_id].extend(cls_idx[start:end].tolist())
            start = end
    return [np.array(ix, dtype=int) for ix in indices]


def partition_iid(y, num_nodes=10, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    chunks = np.array_split(idx, num_nodes)
    return [np.array(c, dtype=int) for c in chunks]


def partition_dirichlet(y, num_nodes=10, alpha=0.5, seed=42):
    """Label-skew Non-IID. Smaller alpha = more heterogeneous."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    props = rng.dirichlet([alpha] * num_nodes, size=len(classes))
    return _assign_by_class(y, num_nodes, props, rng)


def summarize_partition(y, node_indices):
    rows = []
    y = np.asarray(y)
    for i, ix in enumerate(node_indices):
        yy = y[ix]
        rows.append(
            {
                "node": i,
                "n": int(len(ix)),
                "attack_ratio": float(yy.mean()) if len(yy) else 0.0,
                "n_attack": int(yy.sum()) if len(yy) else 0,
                "n_normal": int(len(yy) - yy.sum()) if len(yy) else 0,
            }
        )
    return rows
