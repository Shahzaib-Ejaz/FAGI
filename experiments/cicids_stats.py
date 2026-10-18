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


def ms(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v)), "values": v.tolist()}
