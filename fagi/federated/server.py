import copy
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional


class FederatedServer:
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
        if not client_models:
            return self.global_model

        n_clients = len(client_models)
        if client_weights is None:
            client_weights = [1.0 / n_clients] * n_clients

        total_w = sum(client_weights)
        client_weights = [w / total_w for w in client_weights]

        filtered_models, filtered_weights = self._byzantine_filter(
            client_models, client_weights
        )

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

        if self.dp_mechanism is not None:
            self.dp_mechanism.clip_and_noise(self.global_model)

        return self.global_model

    def _byzantine_filter(
        self,
        models: List[nn.Module],
        weights: List[float],
    ):
        if len(models) < 3:
            return models, weights

        param_vectors = []
        for model in models:
            flat = torch.cat([
                p.data.flatten().float()
                for p in model.parameters()
            ])
            param_vectors.append(flat)

        stacked = torch.stack(param_vectors)
        mean_vec = stacked.mean(dim=0)
        std_val = stacked.std(dim=0).mean().item() + 1e-8

        deviations = (stacked - mean_vec).norm(dim=1)
        threshold = self.byzantine_threshold * std_val * (stacked.shape[1] ** 0.5)

        kept = [(m, w) for m, w, d in zip(models, weights, deviations)
                if d.item() <= threshold]

        if not kept:
            return models, weights

        filtered_models, filtered_weights = zip(*kept)

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
