"""Dense-adjacency GraphSAGE used by the experiments.

Every graph in both evaluation datasets has a fixed node count (12 endpoint
nodes for API sessions, k+1 flow nodes for CICIDS2017 ego graphs), so graphs
are stored as dense tensors X[B, n, d] and A[B, n, n]. This makes batched PGD
and per-sample gradients (needed for DP-SGD) cheap and exact.

DenseSAGEConv computes lin_l(mean_{j in N(i)} h_j) + lin_r(h_i), the same
function as torch_geometric's SAGEConv(aggr="mean"); A[i, j] counts edges
j -> i so duplicate edges are weighted exactly as in the sparse version.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DenseSAGEConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_l = nn.Linear(in_dim, out_dim)
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h: Tensor, A: Tensor) -> Tensor:
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1.0)
        agg = (A @ h) / deg
        return self.lin_l(agg) + self.lin_r(h)


def dense_gnnguard(x: Tensor, A: Tensor, threshold: float = 0.15) -> Tensor:
    """Drop edges whose endpoint features have cosine similarity < threshold.

    Mirrors fagi.model.gnnguard.gnnguard_filter: if every edge of a graph
    would be removed, that graph keeps its original edges.
    """
    xn = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = xn @ xn.transpose(-1, -2)
    pruned = A * (sim >= threshold).to(A.dtype)
    emptied = (pruned.sum(dim=(-1, -2)) == 0) & (A.sum(dim=(-1, -2)) > 0)
    return torch.where(emptied[..., None, None], A, pruned)


class DenseFAGI(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple = (64, 32),
        n_classes: int = 2,
        dropout: float = 0.25,
        gnnguard_threshold: Optional[float] = 0.15,
        readout: str = "mean",
    ):
        super().__init__()
        if readout not in ("mean", "center"):
            raise ValueError("readout must be 'mean' or 'center'")
        h1, h2 = hidden_dims
        self.conv1 = DenseSAGEConv(in_dim, h1)
        self.conv2 = DenseSAGEConv(h1, h2)
        self.fc = nn.Linear(h2, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.gnnguard_threshold = gnnguard_threshold
        self.readout = readout

    def forward(self, x: Tensor, A: Tensor) -> Tensor:
        if self.gnnguard_threshold is not None:
            A = dense_gnnguard(x, A, self.gnnguard_threshold)
        h = F.relu(self.conv1(x, A))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, A))
        z = h.mean(dim=-2) if self.readout == "mean" else h[..., 0, :]
        return self.fc(z)


def pgd_attack_dense(
    model: nn.Module,
    X: Tensor,
    A: Tensor,
    Y: Tensor,
    epsilon: float = 0.1,
    steps: int = 10,
    alpha: Optional[float] = None,
    node_mask: Optional[Tensor] = None,
    random_start: bool = True,
) -> Tensor:
    """L-infinity PGD on node features (Madry et al., 2018).

    node_mask[B, n] restricts the perturbation to nodes the attacker controls.
    The step size defaults to 2.5 * epsilon / steps.
    """
    if alpha is None:
        alpha = 2.5 * epsilon / steps
    was_training = model.training
    model.eval()
    m = node_mask.unsqueeze(-1).to(X.dtype) if node_mask is not None else torch.ones_like(X)
    if random_start:
        delta = torch.empty_like(X).uniform_(-epsilon, epsilon) * m
    else:
        delta = torch.zeros_like(X)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(X + delta, A), Y)
        (g,) = torch.autograd.grad(loss, delta)
        delta = ((delta + alpha * g.sign()).clamp(-epsilon, epsilon) * m).detach()
    model.train(was_training)
    return (X + delta).detach()
