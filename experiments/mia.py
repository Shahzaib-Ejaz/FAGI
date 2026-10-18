from typing import Dict, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _tpr_at(fpr, tpr, target):
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def mia_report(loss_members: np.ndarray, loss_nonmembers: np.ndarray, labels_members: Optional[np.ndarray] = None,
               labels_nonmembers: Optional[np.ndarray] = None, seed: int = 0, min_class: int = 20) -> Dict:
    rng = np.random.RandomState(seed)
    n = min(len(loss_members), len(loss_nonmembers))
    im = rng.choice(len(loss_members), n, replace=False)
    inn = rng.choice(len(loss_nonmembers), n, replace=False)
    lm, ln = loss_members[im], loss_nonmembers[inn]
    scores = -np.concatenate([lm, ln])
    truth = np.concatenate([np.ones(n), np.zeros(n)])
    fpr, tpr, _ = roc_curve(truth, scores)
    half = n // 2
    cal = np.concatenate([np.arange(half), n + np.arange(half)])
    ev = np.concatenate([np.arange(half, n), n + np.arange(half, n)])
    best_t, best_acc = None, -1.0
    for t in np.quantile(scores[cal], np.linspace(0, 1, 401)):
        acc = ((scores[cal] >= t) == truth[cal]).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    out = {"auc": float(roc_auc_score(truth, scores)),
           "attack_accuracy": float(((scores[ev] >= best_t) == truth[ev]).mean()),
           "tpr_at_1pct_fpr": _tpr_at(fpr, tpr, 0.01), "tpr_at_0.1pct_fpr": _tpr_at(fpr, tpr, 0.001),
           "n_pairs": int(n)}
    if labels_members is not None and labels_nonmembers is not None:
        per = {}
        for lab in np.unique(labels_members):
            m = loss_members[labels_members == lab]
            nn = loss_nonmembers[labels_nonmembers == lab]
            if len(m) >= min_class and len(nn) >= min_class:
                k = min(len(m), len(nn))
                s = -np.concatenate([rng.choice(m, k, replace=False), rng.choice(nn, k, replace=False)])
                per[str(lab)] = float(roc_auc_score(np.concatenate([np.ones(k), np.zeros(k)]), s))
        out["per_class_auc"] = per
        out["max_per_class_auc"] = max(per.values()) if per else None
    return out
