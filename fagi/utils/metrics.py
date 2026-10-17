import numpy as np
from typing import List, Tuple


def detection_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    benign_mask = y_true == 0
    if benign_mask.sum() == 0:
        return 0.0
    fp = np.sum((y_pred == 1) & benign_mask)
    tn = np.sum((y_pred == 0) & benign_mask)
    return float(fp / (fp + tn + 1e-10))


def detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    attack_mask = y_true == 1
    if attack_mask.sum() == 0:
        return 0.0
    tp = np.sum((y_pred == 1) & attack_mask)
    fn = np.sum((y_pred == 0) & attack_mask)
    return float(tp / (tp + fn + 1e-10))


def precision_attack(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return float(tp / (tp + fp + 1e-10))


def full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    acc = detection_accuracy(y_true, y_pred)
    fpr = false_positive_rate(y_true, y_pred)
    dr  = detection_rate(y_true, y_pred)
    pr  = precision_attack(y_true, y_pred)
    f1  = 2 * pr * dr / (pr + dr + 1e-10)

    report = dict(accuracy=acc, fpr=fpr, detection_rate=dr,
                  precision=pr, f1=f1)

    if label:
        print(f"\n{'='*40}")
        print(f"  {label}")
        print(f"{'='*40}")
        print(f"  Accuracy:       {acc*100:.2f}%")
        print(f"  FPR:            {fpr*100:.2f}%")
        print(f"  Detection Rate: {dr*100:.2f}%")
        print(f"  Precision:      {pr*100:.2f}%")
        print(f"  F1:             {f1*100:.2f}%")

    return report


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Tuple[float, float]:
    from scipy.stats import chi2

    b = np.sum((pred_a == y_true) & (pred_b != y_true))
    c = np.sum((pred_a != y_true) & (pred_b == y_true))

    if b + c == 0:
        return 0.0, 1.0

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return float(chi2_stat), float(p_value)
