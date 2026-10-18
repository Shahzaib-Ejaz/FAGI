import random
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fagi.model.dense import pgd_attack_dense
from fagi.privacy.dp_sgd import DPSGD, make_graph_loss

PGD_EPS = 0.1
PGD_TRAIN_STEPS = 10
PGD_EVAL_STEPS = 20
DP_SIGMA = 1.1
DP_CLIP = 1.0
DP_DELTA = 1e-5


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def T(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype)


def train_model(
    model: nn.Module,
    X: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    node_mask: Optional[np.ndarray] = None,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    adv: bool = False,
    dp: bool = False,
    sigma: float = DP_SIGMA,
    clip: float = DP_CLIP,
    delta: float = DP_DELTA,
    seed: int = 0,
    pgd_eps: float = PGD_EPS,
    pgd_steps: int = PGD_TRAIN_STEPS,
    lambda_adv: float = 0.5,
) -> Dict:
    gen = set_seed(seed)
    Xt, At, Yt = T(X), T(A), T(Y, torch.long)
    Mt = T(node_mask) if node_mask is not None else None
    n = len(Yt)
    t0 = time.time()
    if dp:
        opt = DPSGD(model, n, batch_size, sigma=sigma, clip_norm=clip, lr=lr,
                    weight_decay=weight_decay, delta=delta, generator=gen)
        loss_fn = make_graph_loss(model, lambda_adv if adv else None)
        n_steps = int(np.ceil(epochs / opt.sample_rate))
        for _ in range(n_steps):
            idx = opt.poisson_batch()
            if len(idx) == 0:
                opt.apply(opt.clipped_noisy_grad(None))
                continue
            xb, ab, yb = Xt[idx], At[idx], Yt[idx]
            model.train()
            if adv:
                xa = pgd_attack_dense(model, xb, ab, yb, pgd_eps, pgd_steps,
                                      node_mask=Mt[idx] if Mt is not None else None)
                model.train()
                pg = opt.per_sample_grads(loss_fn, xb, ab, yb, xa)
            else:
                pg = opt.per_sample_grads(loss_fn, xb, ab, yb)
            opt.apply(opt.clipped_noisy_grad(pg))
        info = {"dp": True, "sigma": sigma, "clip": clip, "delta": delta,
                "sample_rate": opt.sample_rate, "steps": opt.steps, "epsilon": opt.epsilon()}
    else:
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            perm = torch.randperm(n, generator=gen)
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                xb, ab, yb = Xt[idx], At[idx], Yt[idx]
                if adv:
                    xa = pgd_attack_dense(model, xb, ab, yb, pgd_eps, pgd_steps,
                                          node_mask=Mt[idx] if Mt is not None else None)
                model.train()
                optim.zero_grad()
                loss = F.cross_entropy(model(xb, ab), yb)
                if adv:
                    loss = lambda_adv * loss + (1 - lambda_adv) * F.cross_entropy(model(xa, ab), yb)
                loss.backward()
                optim.step()
        info = {"dp": False, "epsilon": None}
    info["train_seconds"] = round(time.time() - t0, 1)
    model.eval()
    return info


@torch.no_grad()
def predict(model: nn.Module, X, A, batch: int = 2048) -> np.ndarray:
    model.eval()
    Xt, At = T(X), T(A)
    out = [model(Xt[i:i + batch], At[i:i + batch]).argmax(1) for i in range(0, len(Xt), batch)]
    return torch.cat(out).numpy()


@torch.no_grad()
def per_sample_loss(model: nn.Module, X, A, Y, batch: int = 2048) -> np.ndarray:
    model.eval()
    Xt, At, Yt = T(X), T(A), T(Y, torch.long)
    out = [F.cross_entropy(model(Xt[i:i + batch], At[i:i + batch]), Yt[i:i + batch], reduction="none")
           for i in range(0, len(Xt), batch)]
    return torch.cat(out).numpy()


def adversarial_inputs(model, X, A, Y, node_mask=None, eps=PGD_EPS, steps=PGD_EVAL_STEPS,
                       batch: int = 1024, seed: int = 0) -> np.ndarray:
    set_seed(seed)
    out = []
    for i in range(0, len(X), batch):
        m = T(node_mask[i:i + batch]) if node_mask is not None else None
        out.append(pgd_attack_dense(model, T(X[i:i + batch]), T(A[i:i + batch]),
                                    T(Y[i:i + batch], torch.long), eps, steps, node_mask=m).numpy())
    return np.concatenate(out)
