"""
Projected Gradient Descent (PGD) Adversarial Training
======================================================
Implements the PGD attack and adversarial training protocol from:

    Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A.
    (2018). Towards Deep Learning Models Resistant to Adversarial
    Attacks. ICLR 2018.

FAGI uses PGD to harden the GNN detector against feature-space
perturbations of API call graph node attributes.

Training objective (equal weighting as in Madry et al.):
    L = 0.5 × L_ce(f_W(G), y) + 0.5 × L_ce(f_W(G + δ*), y)

where δ* = argmax_{‖δ‖_∞ ≤ ε} L_ce(f_W(G + δ), y)
approximated by K_pgd steps of projected gradient ascent.

Paper hyperparameters:
    ε = 0.1  (L-infinity perturbation budget)
    K = 10   (PGD steps)
    α = 0.01 (step size)
"""

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
    """
    PGD attack on graph node features (L-infinity norm).

    Finds adversarial perturbation δ* such that:
        ‖δ*‖_∞ ≤ ε  and  δ* ≈ argmax L_ce(f_W(x + δ), y)

    Parameters
    ----------
    model : nn.Module
        The GNN model to attack (in eval mode).
    x : Tensor, shape (n_nodes, d)
        Clean node feature matrix.
    edge_index : Tensor, shape (2, n_edges)
        Graph connectivity.
    y : Tensor, shape (1,)
        True label.
    epsilon : float
        L-infinity perturbation budget ε.
    steps : int
        Number of PGD iterations K.
    alpha : float
        Step size α per iteration.
    random_start : bool
        Whether to initialise δ with random uniform noise in [-ε, ε].

    Returns
    -------
    Tensor, shape (n_nodes, d)
        Adversarially perturbed node features x + δ*.
    """
    model.eval()

    # Initialise perturbation
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
            # Gradient ascent step (maximise loss)
            x_adv = x_adv + alpha * x_adv.grad.sign()
            # Project back into L-infinity ball around x
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
    """
    Compute combined clean + adversarial loss for PGD training.

    L = lambda_adv × L_ce(clean) + (1 - lambda_adv) × L_ce(adversarial)

    Following Madry et al., lambda_adv = 0.5 gives equal weighting.

    Parameters
    ----------
    model : nn.Module
        GNN model (in train mode).
    x, edge_index, y :
        Clean graph and label.
    epsilon : float
        PGD perturbation budget.
    pgd_steps : int
        PGD attack steps during training.
    alpha : float
        PGD step size.
    lambda_adv : float
        Weight on clean loss (default 0.5 = equal weighting).

    Returns
    -------
    Tensor (scalar)
        Combined adversarial training loss.
    """
    # Compute adversarial examples — pgd_attack manages its own gradient context
    x_adv = pgd_attack(
        model, x, edge_index, y,
        epsilon=epsilon, steps=pgd_steps, alpha=alpha,
    )
    x_adv = x_adv.detach()  # detach so clean/adv loss are independent

    # Clean loss
    out_clean = model(x, edge_index)
    loss_clean = F.cross_entropy(out_clean, y)

    # Adversarial loss
    out_adv = model(x_adv, edge_index)
    loss_adv = F.cross_entropy(out_adv, y)

    return lambda_adv * loss_clean + (1.0 - lambda_adv) * loss_adv
