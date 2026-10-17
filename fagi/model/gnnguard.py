import torch
from torch import Tensor


def gnnguard_filter(
    x: Tensor,
    edge_index: Tensor,
    threshold: float = 0.15,
) -> Tensor:
    src, dst = edge_index[0], edge_index[1]

    if len(src) == 0:
        return edge_index

    x_src = x[src]
    x_dst = x[dst]

    norm_src = x_src.norm(dim=1, keepdim=True).clamp(min=1e-8)
    norm_dst = x_dst.norm(dim=1, keepdim=True).clamp(min=1e-8)

    similarity = ((x_src / norm_src) * (x_dst / norm_dst)).sum(dim=1)

    mask = similarity >= threshold

    if mask.sum() == 0:
        return edge_index

    return edge_index[:, mask]
