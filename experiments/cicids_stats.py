import numpy as np
from scipy.stats import wilcoxon


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    if np.allclose(diff, 0):
        return {"mean_diff": 0.0, "p": 1.0, "n": int(len(a)), "wins_a": 0, "wins_b": 0}
    stat, p = wilcoxon(a, b)
    return {"mean_diff": float(diff.mean()), "p": float(p), "n": int(len(a)),
            "wins_a": int((diff > 0).sum()), "wins_b": int((diff < 0).sum())}


def paired_folds(a, b, n_folds):
    """Paired Wilcoxon test on fold means.

    The seeds of a fold share its test set, private/pool split and attacked flows, so fold-seed pairs are not
    independent; the fold is the unit. With five folds the smallest attainable two-sided p is 0.0625, so the
    win counts carry the result. pair_wins_* count the individual fold-seed pairs for reference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    fa, fb = a.reshape(n_folds, -1).mean(axis=1), b.reshape(n_folds, -1).mean(axis=1)
    out = paired(fa, fb)
    out.update({"unit": "fold (seeds averaged)", "pairs": int(len(a)), "pair_wins_a": int((a - b > 0).sum()),
                "pair_wins_b": int((a - b < 0).sum())})
    return out


def ms(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v)), "values": v.tolist()}
