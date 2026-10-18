from typing import List, Tuple

import numpy as np

from fagi.graph.constructor import APIGraphConstructor


def sessions_to_dense(
    sessions: List[dict],
    include_payload: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    constructor = APIGraphConstructor(include_payload=include_payload)
    n = constructor.n_nodes
    X = np.zeros((len(sessions), n, constructor.feature_dim), dtype=np.float32)
    A = np.zeros((len(sessions), n, n), dtype=np.float32)
    y = np.zeros(len(sessions), dtype=np.int64)
    for i, s in enumerate(sessions):
        g = constructor.session_to_graph(s)
        X[i] = g.x.numpy()
        ei = g.edge_index.numpy()
        np.add.at(A[i], (ei[1], ei[0]), 1.0)
        y[i] = int(g.y.item())
    mask = (X[:, :, 0] > 0).astype(np.float32)
    return X, A, mask, y


class NodeFeatureScaler:
    def fit(self, X: np.ndarray, mask: np.ndarray) -> "NodeFeatureScaler":
        active = X[mask > 0]
        self.scale_ = active.std(axis=0) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X / self.scale_).astype(np.float32)
