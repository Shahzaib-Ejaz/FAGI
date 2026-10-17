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
    ) -> float:
        import math
        q = 1.0
        alpha = 10
        rdp = alpha * q**2 / (2 * sigma**2) * n_rounds
        epsilon = rdp + math.log(1 / delta) / (alpha - 1)
        return epsilon
