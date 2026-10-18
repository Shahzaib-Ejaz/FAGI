import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class MVGMechanism:
    def __init__(
        self,
        clip_norm: float = 1.0,
        sigma: float = 1.1,
        device: str = "cpu",
    ):
        self.clip_norm = clip_norm
        self.sigma = sigma
        self.device = device

    def clip_and_noise(self, model: nn.Module) -> None:
        nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)

        noise_scale = self.sigma * self.clip_norm
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * noise_scale
                    param.grad += noise

    @staticmethod
    def compute_epsilon(
        sigma: float,
        clip_norm: float,
        n_rounds: int,
        delta: float = 1e-5,
        sample_rate: float = 1.0,
    ) -> float:
        from fagi.privacy.dp_sgd import compute_epsilon
        return compute_epsilon(sigma, sample_rate, n_rounds, delta)
