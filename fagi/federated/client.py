"""
Federated Client — Per-Organisation Local Training
====================================================
Each organisational node trains a local copy of the global model
on its private API call graph data, then sends only the updated
model weights (never raw logs) to the aggregation server.
"""

import copy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Optional

from fagi.model.pgd import adversarial_loss
from fagi.privacy.mvg_dp import MVGMechanism


class FederatedClient:
    """
    Local trainer for one federated organisational node.

    Parameters
    ----------
    org_id : str
        Organisation identifier (for logging).
    local_data : list[Data]
        Local API call graph dataset (private, never shared).
    dp_mechanism : MVGMechanism, optional
        If provided, applies DP noise after each gradient step.
    adversarial_training : bool
        Whether to use PGD adversarial training (recommended: True).
    pgd_epsilon : float
        PGD perturbation budget.
    pgd_steps : int
        PGD attack steps per training iteration.
    """

    def __init__(
        self,
        org_id: str,
        local_data: List[Data],
        dp_mechanism: Optional[MVGMechanism] = None,
        adversarial_training: bool = True,
        pgd_epsilon: float = 0.1,
        pgd_steps: int = 10,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
    ):
        self.org_id = org_id
        self.local_data = local_data
        self.dp_mechanism = dp_mechanism
        self.adversarial_training = adversarial_training
        self.pgd_epsilon = pgd_epsilon
        self.pgd_steps = pgd_steps
        self.lr = lr
        self.weight_decay = weight_decay

    def local_train(
        self,
        global_model: torch.nn.Module,
        epochs: int = 5,
        batch_size: int = 200,
        verbose: bool = False,
    ) -> torch.nn.Module:
        """
        Train a local copy of the global model on this node's private data.

        Parameters
        ----------
        global_model : nn.Module
            Current global model weights (downloaded from server).
        epochs : int
            Number of local training epochs.
        batch_size : int
            Number of graphs per gradient step.
        verbose : bool
            Print per-epoch loss.

        Returns
        -------
        nn.Module
            Locally fine-tuned model (sent back to server for aggregation).
        """
        local_model = copy.deepcopy(global_model)
        optimizer = torch.optim.Adam(
            local_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        n = len(self.local_data)
        import numpy as np

        for epoch in range(epochs):
            local_model.train()
            indices = np.random.permutation(n)[:min(batch_size, n)]
            epoch_loss = 0.0

            for idx in indices:
                g = self.local_data[idx]
                optimizer.zero_grad()

                if self.adversarial_training and epoch % 2 == 0:
                    loss = adversarial_loss(
                        local_model, g.x, g.edge_index, g.y,
                        epsilon=self.pgd_epsilon,
                        pgd_steps=self.pgd_steps,
                    )
                else:
                    out = local_model(g.x, g.edge_index)
                    loss = F.cross_entropy(out, g.y)

                loss.backward()

                # Apply DP noise after backprop
                if self.dp_mechanism is not None:
                    self.dp_mechanism.clip_and_noise(local_model)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        local_model.parameters(), 1.0
                    )

                optimizer.step()
                epoch_loss += loss.item()

            if verbose:
                avg_loss = epoch_loss / max(len(indices), 1)
                print(f"  Org {self.org_id} | Epoch {epoch+1}/{epochs} "
                      f"| Loss: {avg_loss:.4f}")

        return local_model
