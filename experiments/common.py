import json
import os
import random
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fagi.model.dense import DenseFAGI, pgd_attack_dense
from fagi.privacy.dp_sgd import compute_epsilon
from fagi.training import (DP_CLIP, DP_DELTA, DP_SIGMA, PGD_EPS, PGD_EVAL_STEPS, PGD_TRAIN_STEPS, T,
                           adversarial_inputs, per_sample_loss, predict, set_seed, train_model)
from fagi.utils.metrics import full_report, mcnemar_test

torch.set_num_threads(max(1, os.cpu_count() or 1))



class FlatWrapper(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x, A):
        return self.net(x[..., 0, :])


class MLP(nn.Module):
    def __init__(self, d, n_classes=2):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.25),
                               nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.f(x)


class CNN1D(nn.Module):
    def __init__(self, d, n_classes=2):
        super().__init__()
        self.c = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
                               nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
                               nn.AdaptiveMaxPool1d(8))
        self.h = nn.Sequential(nn.Flatten(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.h(self.c(x.unsqueeze(-2)))


class LSTMNet(nn.Module):
    def __init__(self, d, n_classes=2, hidden=32):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        _, (h, _) = self.rnn(x.unsqueeze(-1))
        return self.fc(h[-1])


def flat_model(kind: str, d: int) -> nn.Module:
    return FlatWrapper({"mlp": MLP, "cnn": CNN1D, "lstm": LSTMNet}[kind](d))


def as_graph(X2d: np.ndarray):
    return X2d[:, None, :].astype(np.float32), np.zeros((len(X2d), 1, 1), dtype=np.float32)


def metrics(y, pred) -> Dict:
    r = full_report(np.asarray(y), np.asarray(pred))
    return {k: float(v) for k, v in r.items()}


def per_class_detection(labels: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    out = {}
    for lab in sorted(set(labels)):
        m = labels == lab
        if lab == "BENIGN":
            out[lab] = float((pred[m] == 0).mean())
        else:
            out[lab] = float((pred[m] == 1).mean())
    return out


def membership_inference(loss_members: np.ndarray, loss_nonmembers: np.ndarray, seed: int = 0) -> Dict:
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    n = min(len(loss_members), len(loss_nonmembers))
    lm = rng.choice(loss_members, n, replace=False)
    ln = rng.choice(loss_nonmembers, n, replace=False)
    scores = -np.concatenate([lm, ln])
    truth = np.concatenate([np.ones(n), np.zeros(n)])
    auc = float(roc_auc_score(truth, scores))
    half = n // 2
    cal = np.concatenate([np.arange(half), n + np.arange(half)])
    ev = np.concatenate([np.arange(half, n), n + np.arange(half, n)])
    cands = np.unique(scores[cal])
    best_t, best_acc = cands[0], 0.0
    for t in np.quantile(cands, np.linspace(0, 1, 201)):
        acc = ((scores[cal] >= t) == truth[cal]).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    held_acc = float(((scores[ev] >= best_t) == truth[ev]).mean())
    return {"auc": auc, "attack_accuracy": held_acc, "n_pairs": int(n)}


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "folds": [float(v) for v in vals]}


def mcnemar(y, a, b) -> Dict:
    chi2, p = mcnemar_test(np.asarray(y), np.asarray(a), np.asarray(b))
    return {"chi2": float(chi2), "p": float(p)}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print("saved", path)


def env_info() -> Dict:
    import platform
    import sklearn
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "sklearn": sklearn.__version__,
            "threads": torch.get_num_threads()}
