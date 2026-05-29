"""
Federated Aggregation Server
==============================
Implements the FedAvg aggregation protocol with:
  - Byzantine-robust gradient filtering (3σ rule)
  - MVG differential privacy noise injection
  - Convergence monitoring

Based on:
    McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A.
    (2017). Communication-Efficient Learning of Deep Networks from
    Decentralized Data. AISTATS 2017, pp. 1273–1282.

Paper configuration:
    K = 50 organisational nodes
    T_comm = 60 minutes per round
    Byzantine filter: exclude submissions > 3σ from round mean
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional


class FederatedServer:
    """
    FedAvg aggregation server with Byzantine robustness and DP.

    Parameters
    ----------
    global_model : nn.Module
        The shared global model whose weights are aggregated.
    dp_mechanism : MVGMechanism, optional
        If provided, applies DP noise to the aggregated update.
    byzantine_threshold : float
        Gradient submissions more than `byzantine_threshold` standard
        deviations from the round mean are excluded.
        Paper value: 3.0 (i.e., 3σ filter).
    """

    def __init__(
        self,
        global_model: nn.Module,
        dp_mechanism=None,
        byzantine_threshold: float = 3.0,
    ):
        self.global_model = global_model
        self.dp_mechanism = dp_mechanism
        self.byzantine_threshold = byzantine_threshold
        self.round_history: List[Dict] = []

    def aggregate(
        self,
        client_models: List[nn.Module],
        client_weights: Optional[List[float]] = None,
    ) -> nn.Module:
        """
        FedAvg aggregation with Byzantine filtering.

        Parameters
        ----------
        client_models : list[nn.Module]
            List of locally trained client models.
        client_weights : list[float], optional
            Per-client weights (e.g., proportional to local data size).
            If None, uniform averaging is used.

        Returns
        -------
        nn.Module
            Updated global model.
        """
        if not client_models:
            return self.global_model

        n_clients = len(client_models)
        if client_weights is None:
            client_weights = [1.0 / n_clients] * n_clients

        # Normalise weights
        total_w = sum(client_weights)
        client_weights = [w / total_w for w in client_weights]

        # Byzantine filter: compute per-parameter norms, exclude outliers
        filtered_models, filtered_weights = self._byzantine_filter(
            client_models, client_weights
        )

        # FedAvg: weighted average of model parameters
        global_state = self.global_model.state_dict()
        averaged_state = {}

        with torch.no_grad():
            for param_name in global_state:
                averaged_state[param_name] = torch.zeros_like(
                    global_state[param_name], dtype=torch.float32
                )
                for model, weight in zip(filtered_models, filtered_weights):
                    client_state = model.state_dict()
                    averaged_state[param_name] += (
                        weight * client_state[param_name].float()
                    )

        self.global_model.load_state_dict(averaged_state)

        # Apply DP noise to the aggregated model if mechanism provided
        if self.dp_mechanism is not None:
            self.dp_mechanism.clip_and_noise(self.global_model)

        return self.global_model

    def _byzantine_filter(
        self,
        models: List[nn.Module],
        weights: List[float],
    ):
        """
        Exclude client updates more than `byzantine_threshold` σ
        from the round mean (measured in flattened parameter space).
        """
        # Flatten each model's parameters into a single vector
        param_vectors = []
        for model in models:
            flat = torch.cat([
                p.data.flatten().float()
                for p in model.parameters()
            ])
            param_vectors.append(flat)

        stacked = torch.stack(param_vectors)  # (n_clients, n_params)
        mean_vec = stacked.mean(dim=0)
        std_val = stacked.std(dim=0).mean().item() + 1e-8

        # Compute per-client deviation from mean
        deviations = (stacked - mean_vec).norm(dim=1)
        threshold = self.byzantine_threshold * std_val * (stacked.shape[1] ** 0.5)

        # Keep clients within threshold
        kept = [(m, w) for m, w, d in zip(models, weights, deviations)
                if d.item() <= threshold]

        if not kept:
            # Fallback: keep all if everything is flagged
            return models, weights

        filtered_models, filtered_weights = zip(*kept)

        # Re-normalise weights
        total = sum(filtered_weights)
        filtered_weights = [w / total for w in filtered_weights]

        n_excluded = n_clients_orig = len(models)
        n_excluded -= len(filtered_models)
        if n_excluded > 0:
            print(f"[FedServer] Byzantine filter excluded "
                  f"{n_excluded}/{n_clients_orig} clients.")

        return list(filtered_models), list(filtered_weights)

    def convergence_check(
        self,
        prev_model: nn.Module,
        tau: float = 1e-4,
    ) -> bool:
        """
        Check convergence by comparing current and previous global weights.
        Returns True if ‖W(r+1) − W(r)‖ < τ.
        """
        current_flat = torch.cat([
            p.data.flatten().float()
            for p in self.global_model.parameters()
        ])
        prev_flat = torch.cat([
            p.data.flatten().float()
            for p in prev_model.parameters()
        ])
        delta_norm = (current_flat - prev_flat).norm().item()
        return delta_norm < tau
