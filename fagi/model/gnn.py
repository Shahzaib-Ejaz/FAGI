import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch import Tensor
from typing import Optional

from fagi.model.gnnguard import gnnguard_filter


class FAGIModel(nn.Module):
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
        if self.gnnguard_threshold is not None:
            edge_index = gnnguard_filter(
                x, edge_index, threshold=self.gnnguard_threshold
            )

        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))

        x = x.mean(dim=0, keepdim=True)

        return self.fc(x)

    def predict(self, x: Tensor, edge_index: Tensor) -> int:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return logits.argmax(dim=1).item()
