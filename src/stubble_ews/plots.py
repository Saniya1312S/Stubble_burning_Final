"""Figures used in the README / reports (matplotlib, headless)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from .evaluate import fbeta

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


WATERMARK = ""        # set to e.g. "SYNTHETIC DEMO DATA" by the pipeline when the data is not real


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    if WATERMARK:
        fig.text(0.5, 0.5, WATERMARK, ha="center", va="center", rotation=22, fontsize=20,
                 color="grey", alpha=0.28, weight="bold")
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def pr_curves(y, scores: dict, baseline_point, out_dir):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, lw=2 if name == "ensemble" else 1.1, label=name)
    ax.axhline(np.mean(y), ls=":", c="grey", label=f"prevalence = {np.mean(y):.2f}")
    if baseline_point:
        ax.scatter([baseline_point[1]], [baseline_point[0]], c="k", marker="x", zorder=5, label="persistence rule")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (test set)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "pr_curve_test.png")


def roc_curves(y, scores: dict, out_dir):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=2 if name == "ensemble" else 1.1, label=name)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC (test set)")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "roc_curve_test.png")


def threshold_tradeoff(y_val, p_val, th, beta, out_dir):
    p, r, t = precision_recall_curve(y_val, p_val)
    p, r = p[:-1], r[:-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(t, p, label="precision"); ax.plot(t, r, label="recall"); ax.plot(t, fbeta(p, r, beta), label=f"F{beta:g}")
    for k, c in (("high", "crimson"), ("medium", "darkorange"), ("low", "steelblue")):
        ax.axvline(th[k], c=c, ls="--", lw=1, label=f"{k} = {th[k]:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Score", title="Threshold trade-off (validation set)", xlim=(0, 1))
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, out_dir, "threshold_tradeoff_val.png")


def confusion(cm: dict, out_dir):
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.grid(False)
    ax.imshow(np.log1p(m), cmap="Reds")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=11,
                    color="white" if np.log1p(m[i, j]) > 0.6 * np.log1p(m.max()) else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no alert", "alert"], yticklabels=["no fire", "fire"],
           xlabel="Predicted", ylabel="Actual (next 15 days)", title="Confusion matrix (test, High threshold)")
    return _save(fig, out_dir, "confusion_matrix_test.png")


def feature_importance(names, importances: dict, out_dir):
    avg = np.mean([v / v.sum() for v in importances.values()], axis=0)
    order = np.argsort(avg)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.barh(np.array(names)[order], avg[order], color="#d9534f")
    ax.set(xlabel="Normalised importance (mean of RF & XGBoost)", title="Feature importance")
    return _save(fig, out_dir, "feature_importance.png")


def daily_alerts(dates, y, alert, out_dir):
    import pandas as pd
    d = pd.DataFrame({"date": dates, "y": y, "alert": alert}).groupby("date").sum()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(d.index, d["y"], label="cells with a fire in next 15 d (actual)", c="k")
    ax.plot(d.index, d["alert"], label="cells alerted (High)", c="crimson")
    ax.set(ylabel="Number of H3 cells", title="Alert volume vs. actual events (test period)")
    ax.legend(fontsize=8); fig.autofmt_xdate()
    return _save(fig, out_dir, "daily_alerts_test.png")
