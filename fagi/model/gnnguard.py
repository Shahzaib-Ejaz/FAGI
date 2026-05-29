"""
GNNGuard-Inspired Edge Filtering
=================================
Implements the cosine-similarity neighbour importance filter from:

    Zhang, X. & Zitnik, M. (2020). GNNGuard: Defending Graph Neural
    Networks against Adversarial Attacks. NeurIPS 2020, pp. 9263–9275.
    Harvard University.

This implementation is a simplified version that applies cosine
similarity thresholding to prune edges whose endpoint features are
dissimilar — a heuristic for identifying adversarially injected edges.

The original GNNGuard paper uses a more sophisticated importance score
combining feature similarity with degree normalisation. This simplified
variant captures the core defence mechanism and is labelled
"GNNGuard-inspired" throughout the paper.

Sensitivity to threshold τ:
    τ ∈ {0.05, 0.10, 0.15, 0.20, 0.25} → accuracy variation < 1.2pp
    Default τ = 0.15 as reported in the paper.
"""

import torch
from torch import Tensor


def gnnguard_filter(
    x: Tensor,
    edge_index: Tensor,
    threshold: float = 0.15,
) -> Tensor:
    """
    Filter edges where node features have low cosine similarity.

    Edges connecting nodes with cosine similarity < threshold are
    pruned before message passing, suppressing adversarially injected
    structural noise.

    Parameters
    ----------
    x : Tensor, shape (n_nodes, d)
        Node feature matrix.
    edge_index : Tensor, shape (2, n_edges)
        Edge connectivity in COO format.
    threshold : float
        Cosine similarity threshold τ. Edges with sim < τ are removed.
        Paper value: τ = 0.15.

    Returns
    -------
    Tensor, shape (2, n_kept_edges)
        Filtered edge index. If all edges would be pruned, the
        original edge_index is returned unchanged to prevent
        isolated graphs.
    """
    src, dst = edge_index[0], edge_index[1]

    if len(src) == 0:
        return edge_index

    x_src = x[src]  # (n_edges, d)
    x_dst = x[dst]  # (n_edges, d)

    # L2-normalise each feature vector (clamp to avoid division by zero)
    norm_src = x_src.norm(dim=1, keepdim=True).clamp(min=1e-8)
    norm_dst = x_dst.norm(dim=1, keepdim=True).clamp(min=1e-8)

    similarity = ((x_src / norm_src) * (x_dst / norm_dst)).sum(dim=1)

    mask = similarity >= threshold

    # Safety: if no edges survive, return original (prevents empty graph)
    if mask.sum() == 0:
        return edge_index

    return edge_index[:, mask]
