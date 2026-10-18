import math
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from opacus.accountants import RDPAccountant
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

RDP_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [80, 128, 256, 512, 1024]


def compute_epsilon(sigma: float, sample_rate: float, steps: int, delta: float = 1e-5) -> float:
    if steps <= 0:
        return 0.0
    rdp = compute_rdp(q=sample_rate, noise_multiplier=sigma, steps=steps, orders=RDP_ORDERS)
    eps, _ = get_privacy_spent(orders=RDP_ORDERS, rdp=rdp, delta=delta)
    return float(eps)


class DPSGD:
    def __init__(
        self,
        model: nn.Module,
        n_train: int,
        batch_size: int,
        sigma: float = 1.1,
        clip_norm: float = 1.0,
        lr: float = 2e-3,
        weight_decay: float = 0.0,
        delta: float = 1e-5,
        generator: Optional[torch.Generator] = None,
    ):
        if n_train <= 0:
            raise ValueError("n_train must be positive")
        self.model = model
        self.n_train = n_train
        self.sample_rate = min(1.0, batch_size / n_train)
        self.expected_batch = self.sample_rate * n_train
        self.sigma = sigma
        self.clip_norm = clip_norm
        self.delta = delta
        self.generator = generator
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.accountant = RDPAccountant()
        self.steps = 0

    def poisson_batch(self) -> torch.Tensor:
        u = torch.rand(self.n_train, generator=self.generator)
        return torch.nonzero(u < self.sample_rate, as_tuple=False).flatten()

    def per_sample_grads(self, loss_fn: Callable, *batch) -> Dict[str, torch.Tensor]:
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        in_dims = (None,) + (0,) * len(batch)
        return vmap(grad(loss_fn), in_dims=in_dims, randomness="different")(params, *batch)

    def clipped_noisy_grad(self, per_grads: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        names = [k for k, _ in self.model.named_parameters()]
        shapes = {k: v.shape for k, v in self.model.named_parameters()}
        if per_grads is None:
            summed = {k: torch.zeros(shapes[k]) for k in names}
        else:
            flat = torch.cat([per_grads[k].flatten(1) for k in names], dim=1)
            norms = flat.norm(dim=1)
            factor = (self.clip_norm / (norms + 1e-6)).clamp(max=1.0)
            summed = {k: torch.einsum("b,b...->...", factor, per_grads[k]) for k in names}
        out = {}
        for k in names:
            noise = torch.randn(shapes[k], generator=self.generator) * (self.sigma * self.clip_norm)
            out[k] = (summed[k] + noise) / self.expected_batch
        return out

    def apply(self, noisy: Dict[str, torch.Tensor]) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        for k, p in self.model.named_parameters():
            p.grad = noisy[k].to(p.dtype)
        self.optimizer.step()
        self.accountant.step(noise_multiplier=self.sigma, sample_rate=self.sample_rate)
        self.steps += 1

    def epsilon(self) -> float:
        return compute_epsilon(self.sigma, self.sample_rate, self.steps, self.delta)


def make_graph_loss(model: nn.Module, lambda_adv: Optional[float] = None) -> Callable:
    if lambda_adv is None:
        def loss_fn(params, x, a, y):
            out = functional_call(model, params, (x.unsqueeze(0), a.unsqueeze(0)))
            return F.cross_entropy(out, y.unsqueeze(0))
    else:
        def loss_fn(params, x, a, y, x_adv):
            out = functional_call(model, params, (x.unsqueeze(0), a.unsqueeze(0)))
            out_adv = functional_call(model, params, (x_adv.unsqueeze(0), a.unsqueeze(0)))
            return (lambda_adv * F.cross_entropy(out, y.unsqueeze(0))
                    + (1.0 - lambda_adv) * F.cross_entropy(out_adv, y.unsqueeze(0)))
    return loss_fn


def steps_for(epochs: int, sample_rate: float) -> int:
    return int(math.ceil(epochs / sample_rate))
