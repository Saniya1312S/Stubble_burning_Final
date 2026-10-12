"""Metrics for rare-event forecasting + operating-threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)

from .config import Config


def fbeta(p: np.ndarray | float, r: np.ndarray | float, beta: float):
    b2 = beta ** 2
    return (1 + b2) * p * r / np.maximum(b2 * p + r, 1e-12)


def binary_metrics(y: np.ndarray, yhat: np.ndarray, beta: float = 2.0) -> dict:
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum()); tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec,
        "f1": float(fbeta(prec, rec, 1.0)), f"f{beta:g}": float(fbeta(prec, rec, beta)),
        "accuracy": (tp + tn) / max(len(y), 1),
        "specificity": tn / max(tn + fp, 1),
        "alert_rate": (tp + fp) / max(len(y), 1),
    }


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "prevalence": float(np.mean(y)),
        # accuracy of always predicting the majority class: shows why accuracy alone is misleading
        "majority_class_accuracy": float(max(np.mean(y), 1 - np.mean(y))),
    }


def pick_thresholds(y_val: np.ndarray, p_val: np.ndarray, cfg: Config) -> dict:
    """Operating thresholds chosen on the VALIDATION set only.

    High   = threshold maximising F-beta (beta=2 favours recall).
    Medium = highest threshold that still reaches ``medium_recall_target`` recall, but never above
             0.75 x High, so the three bands can never collapse onto each other.
    Low    = half of Medium.
    """
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    prec, rec = prec[:-1], rec[:-1]
    f = fbeta(prec, rec, cfg.beta)
    k = int(np.nanargmax(f))
    high = float(thr[k])
    ok = np.nonzero(rec >= cfg.medium_recall_target)[0]
    cap = 0.75 * high
    medium = float(min(thr[ok].max(), cap)) if len(ok) else cap
    return {"high": high, "medium": medium, "low": medium / 2,
            "val_fbeta_at_high": float(f[k]), "beta": cfg.beta}


def risk_band(p: np.ndarray, th: dict) -> np.ndarray:
    return np.select([p >= th["high"], p >= th["medium"], p >= th["low"]],
                     ["High", "Medium", "Low"], default="None")
