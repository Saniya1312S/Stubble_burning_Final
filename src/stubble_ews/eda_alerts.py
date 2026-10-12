"""Exploratory analysis of the alert table used by the dashboard (``stubble_alerts_latest.csv``)."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import _save


def analyse(csv_path: str, out_dir: str = ".") -> dict:
    df = pd.read_csv(csv_path)
    try:
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y")
    except Exception:
        df["date"] = pd.to_datetime(df["date"])
    fig_dir = os.path.join(out_dir, "reports", "figures")

    bands = df["risk_band"].value_counts()
    flagged = df[df.any_exceeds == 1]
    per_day = df.groupby("date").any_exceeds.sum()
    cell = df.groupby("h3_id").agg(lat=("lat", "first"), lon=("lon", "first"),
                                   max_p=("p_fire_15d", "max"), days_high=("any_exceeds", "sum"))
    summary = {
        "rows": int(len(df)), "cells": int(df.h3_id.nunique()), "parent_cells": int(df.h3_parent.nunique()),
        "date_min": str(df.date.min().date()), "date_max": str(df.date.max().date()),
        "n_days": int(df.date.nunique()),
        "lat_range": [float(df.lat.min()), float(df.lat.max())],
        "lon_range": [float(df.lon.min()), float(df.lon.max())],
        "band_counts": {k: int(v) for k, v in bands.items()},
        "band_pct": {k: round(100 * v / len(df), 2) for k, v in bands.items()},
        "p_min": float(df.p_fire_15d.min()), "p_median": float(df.p_fire_15d.median()),
        "p_max": float(df.p_fire_15d.max()),
        "share_p_below_0_41": float((df.p_fire_15d < 0.41).mean()),
        "high_alert_rows": int(len(flagged)),
        "high_alert_rate_pct": round(100 * len(flagged) / len(df), 3),
        "cells_ever_high": int((cell.days_high > 0).sum()),
        "cells_ever_high_pct": round(100 * (cell.days_high > 0).mean(), 2),
        "high_cells_per_day": {"min": int(per_day.min()), "mean": round(float(per_day.mean()), 1),
                               "max": int(per_day.max())},
    }
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
    with open(os.path.join(out_dir, "reports", "alerts_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    colors = {"Low": "#4575b4", "Medium": "#f39c12", "High": "#d73027"}
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    order = [b for b in ("Low", "Medium", "High") if b in bands.index]
    ax.bar(order, [bands[b] for b in order], color=[colors[b] for b in order]); ax.set_yscale("log")
    for i, b in enumerate(order):
        ax.text(i, bands[b], f"{bands[b]:,}\n({summary['band_pct'][b]}%)", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="cell-days (log scale)", title="Risk-band distribution", ylim=(1, bands.max() * 6))
    _save(fig, fig_dir, "alerts_band_distribution.png")

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.hist(df.p_fire_15d, bins=60, color="#d9534f"); ax.set_yscale("log")
    ax.set(xlabel="p_fire_15d", ylabel="cell-days (log scale)", title="Distribution of predicted probability")
    _save(fig, fig_dir, "alerts_probability_hist.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(per_day.index, per_day.values, marker="o", ms=3, c="#d73027")
    ax.set(ylabel="H3 cells flagged High", title="Daily number of High-risk cells (July 2024 sample)")
    fig.autofmt_xdate()
    _save(fig, fig_dir, "alerts_daily_high.png")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    rest = cell[cell.days_high == 0]; hot = cell[cell.days_high > 0]
    ax.scatter(rest.lon, rest.lat, s=3, c="#c8c8c8", label=f"never High ({len(rest):,} cells)")
    sc = ax.scatter(hot.lon, hot.lat, s=9, c=hot.days_high, cmap="Reds", label=f"High at least once ({len(hot)} cells)")
    plt.colorbar(sc, ax=ax, label="days flagged High")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Spatial footprint of the alert table (H3 res 7)")
    ax.legend(fontsize=8, loc="lower left"); ax.set_aspect("equal")
    _save(fig, fig_dir, "alerts_spatial_footprint.png")
    return summary
