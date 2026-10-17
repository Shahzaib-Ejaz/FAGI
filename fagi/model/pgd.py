import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


def pgd_attack(
    model: torch.nn.Module,
    x: Tensor,
    edge_index: Tensor,
    y: Tensor,
    epsilon: float = 0.1,
    steps: int = 10,
    alpha: float = 0.01,
    random_start: bool = True,
) -> Tensor:
    model.eval()

    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(x)

    x_adv = (x + delta).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        out = model(x_adv, edge_index)
        loss = F.cross_entropy(out, y)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(
                torch.min(x_adv, x + epsilon), x - epsilon
            )
        x_adv = x_adv.detach()

    return x_adv


def adversarial_loss(
    model: torch.nn.Module,
    x: Tensor,
    edge_index: Tensor,
    y: Tensor,
    epsilon: float = 0.1,
    pgd_steps: int = 10,
    alpha: float = 0.01,
    lambda_adv: float = 0.5,
) -> Tensor:
    x_adv = pgd_attack(
        model, x, edge_index, y,
        epsilon=epsilon, steps=pgd_steps, alpha=alpha,
    )
    x_adv = x_adv.detach()

    out_clean = model(x, edge_index)
    loss_clean = F.cross_entropy(out_clean, y)

    out_adv = model(x_adv, edge_index)
    loss_adv = F.cross_entropy(out_adv, y)

    return lambda_adv * loss_clean + (1.0 - lambda_adv) * loss_adv
