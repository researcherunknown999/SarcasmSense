from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


@dataclass
class BinaryThresholds:
    sarc: float = 0.50
    shift: float = 0.50


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def tune_binary_threshold(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    grid: np.ndarray = np.linspace(0.0, 1.0, 101),
) -> Tuple[float, float]:
    """
    Choose threshold that maximizes macro-F1 for a binary task.
    Returns (best_thr, best_macro_f1).
    """
    best_thr = 0.5
    best_f1 = -1.0
    for thr in grid:
        y_pred = (p_pos >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def compute_binary_metrics(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    thr: float,
    with_auc: bool = True,
) -> Dict[str, float]:
    y_pred = (p_pos >= thr).astype(int)
    out = {
        "macro_f1": _safe_float(f1_score(y_true, y_pred, average="macro")),
        "acc": _safe_float(accuracy_score(y_true, y_pred)),
    }
    if with_auc:
        # AUCs require both classes present; guard for edge splits
        try:
            out["roc_auc"] = _safe_float(roc_auc_score(y_true, p_pos))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = _safe_float(average_precision_score(y_true, p_pos))
        except Exception:
            out["pr_auc"] = float("nan")
    return out


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "macro_f1": _safe_float(f1_score(y_true, y_pred, average="macro")),
        "acc": _safe_float(accuracy_score(y_true, y_pred)),
    }


def compute_confusions(
    y_sarc: np.ndarray, y_sarc_pred: np.ndarray,
    y_shift: np.ndarray, y_shift_pred: np.ndarray,
    y_target: np.ndarray, y_target_pred: np.ndarray,
    num_target_classes: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    cm_sarc = confusion_matrix(y_sarc, y_sarc_pred, labels=[0, 1])
    cm_shift = confusion_matrix(y_shift, y_shift_pred, labels=[0, 1])
    if num_target_classes is None:
        labels = np.unique(y_target)
    else:
        labels = list(range(int(num_target_classes)))
    cm_target = confusion_matrix(y_target, y_target_pred, labels=labels)
    return {"sarc": cm_sarc, "shift": cm_shift, "target": cm_target}


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Generic bootstrap CI for a scalar metric function metric_fn(y_true, y_pred)->float.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(np.array(vals, dtype=np.float32), [2.5, 97.5])
    return float(lo), float(hi)
