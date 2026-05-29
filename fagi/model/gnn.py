"""
Adversarial GNN Detector
=========================
GraphSAGE-based graph classifier with GNNGuard-inspired edge filtering
and optional PGD adversarial training.

Architecture:
    GraphSAGE L=2, hidden dims [64, 32], mean aggregation
    → GNNGuard edge filter (τ=0.15)
    → 2-layer MLP classification head
    → Binary output: benign (0) / malicious (1)

Total parameters: ~150K for the default configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch import Tensor
from typing import Optional

from fagi.model.gnnguard import gnnguard_filter


class FAGIModel(nn.Module):
    """
    FAGI GraphSAGE detector with GNNGuard edge filtering.

    Parameters
    ----------
    in_dim : int
        Node feature dimension. 6 with LLM payload features, 5 without.
    hidden_dims : tuple[int, int]
        Hidden dimensions for the two GraphSAGE layers.
        Paper values: (64, 32).
    n_classes : int
        Number of output classes (2 for binary benign/malicious).
    dropout : float
        Dropout probability applied after each SAGEConv layer.
    gnnguard_threshold : float
        Cosine similarity threshold τ for GNNGuard edge filtering.
        Paper value: 0.15. Set to None to disable GNNGuard.
    """

    def __init__(
        self,
        in_dim: int = 6,
        hidden_dims: tuple = (64, 32),
        n_classes: int = 2,
        dropout: float = 0.25,
        gnnguard_threshold: Optional[float] = 0.15,
    ):
        super().__init__()
        h1, h2 = hidden_dims

        self.conv1 = SAGEConv(in_dim, h1, aggr="mean")
        self.conv2 = SAGEConv(h1, h2, aggr="mean")
        self.fc = nn.Linear(h2, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.gnnguard_threshold = gnnguard_threshold

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass through the GNN detector.

        Parameters
        ----------
        x : Tensor, shape (n_nodes, in_dim)
            Node feature matrix.
        edge_index : Tensor, shape (2, n_edges)
            Graph connectivity in COO format.

        Returns
        -------
        Tensor, shape (1, n_classes)
            Graph-level logits (one prediction per graph via mean pooling).
        """
        # GNNGuard: filter edges with low cosine similarity
        if self.gnnguard_threshold is not None:
            edge_index = gnnguard_filter(
                x, edge_index, threshold=self.gnnguard_threshold
            )

        # Message passing
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))

        # Graph-level readout: mean pooling over all nodes
        x = x.mean(dim=0, keepdim=True)  # shape: (1, h2)

        return self.fc(x)

    def predict(self, x: Tensor, edge_index: Tensor) -> int:
        """Return predicted class label (0=benign, 1=malicious)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return logits.argmax(dim=1).item()
