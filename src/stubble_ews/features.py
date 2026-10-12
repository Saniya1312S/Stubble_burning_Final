"""Feature engineering: fire detections -> (H3 cell x day) panel with a 15-day-ahead label.

Leakage rules (enforced + unit-tested)
--------------------------------------
* every feature at day *t* uses detections from days <= t only;
* the label at day *t* = "any detection in days t+1 .. t+horizon";
* rows whose label window runs past the end of the data get ``y = NaN`` (never treated as 0).
"""
from __future__ import annotations

from typing import List, Tuple

import h3
import numpy as np
import pandas as pd
from scipy import sparse

from .config import PUNJAB_BBOX, Config


def make_grid(cfg: Config) -> List[str]:
    """All H3 cells (at ``cfg.h3_res``) whose centre falls inside the Punjab bounding box."""
    b = PUNJAB_BBOX
    outer = [(b["lat_min"], b["lon_min"]), (b["lat_min"], b["lon_max"]),
             (b["lat_max"], b["lon_max"]), (b["lat_max"], b["lon_min"])]
    return sorted(h3.polygon_to_cells(h3.LatLngPoly(outer), cfg.h3_res))


def _neighbour_matrix(cells: List[str], idx: dict) -> sparse.csr_matrix:
    rows, cols = [], []
    for c, i in idx.items():
        for nb in h3.grid_disk(c, 1):
            j = idx.get(nb)
            if j is not None and j != i:
                rows.append(i)
                cols.append(j)
    n = len(cells)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))


def feature_names(cfg: Config) -> List[str]:
    return ([f"fires_{w}d" for w in cfg.windows]
            + ["nb_fires_7d", "nb_fires_30d", "days_since_last_fire", "hist_rate",
               "fires_last_year_window", "doy_sin", "doy_cos", "is_kharif", "lat", "lon"])


def build_panel(firms: pd.DataFrame, cfg: Config, *, season_only: bool = True,
                last_n_days: int | None = None,
                date_range: Tuple[str, str] | None = None) -> Tuple[pd.DataFrame, List[str]]:
    """Build the modelling table.

    Parameters
    ----------
    firms : frame with ``date, latitude, longitude`` (+ optional ``weight``).
    season_only : keep only rows whose month is in ``cfg.season_months``.
    last_n_days : if set, return only the final N calendar days (used to export alerts); overrides season filter.
    date_range : force the panel's first/last day (otherwise min/max date of ``firms``).

    Returns ``(panel, feature_columns)``. Count-like features are ``log1p``-scaled.
    """
    H = cfg.horizon_days
    cell_of = np.array([h3.latlng_to_cell(la, lo, cfg.h3_res)
                        for la, lo in zip(firms.latitude.values, firms.longitude.values)], dtype=object)
    cells = sorted(set(make_grid(cfg)) | set(cell_of.tolist()))
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)

    d0, d1 = (pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])) if date_range else \
             (firms.date.min(), firms.date.max())
    days = pd.date_range(d0, d1, freq="D")
    T = len(days)
    if T <= H + cfg.burn_in_days:
        raise ValueError(f"Need more than {H + cfg.burn_in_days} days of data, got {T}.")

    # --- daily detections per cell: C[cell, day] --------------------------------------------
    C = np.zeros((n, T), dtype=np.float32)
    ci = np.array([idx[c] for c in cell_of])
    di = (firms.date.values.astype("datetime64[D]") - days[0].to_datetime64().astype("datetime64[D]")).astype(int)
    ok = (di >= 0) & (di < T)
    np.add.at(C, (ci[ok], di[ok]), firms.weight.values[ok].astype(np.float32))

    cs = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(C, axis=1, dtype=np.float64).astype(np.float32)], axis=1)
    t = np.arange(T)

    def window_sum(w: int) -> np.ndarray:               # detections in days [t-w+1, t]
        return cs[:, t + 1] - cs[:, np.maximum(0, t + 1 - w)]

    feats = {}
    sums = {w: window_sum(w) for w in sorted(set(cfg.windows) | {7, 30})}
    for w in cfg.windows:
        feats[f"fires_{w}d"] = np.log1p(sums[w])

    A = _neighbour_matrix(cells, idx)
    feats["nb_fires_7d"] = np.log1p(np.asarray(A @ sums[7]))
    feats["nb_fires_30d"] = np.log1p(np.asarray(A @ sums[30]))

    last = np.full(n, -366)                              # days since the last detection (capped at 365)
    dsl = np.empty((n, T), dtype=np.float32)
    for k in range(T):
        last = np.where(C[:, k] > 0, k, last)
        dsl[:, k] = np.minimum(k - last, 365)
    feats["days_since_last_fire"] = dsl

    feats["hist_rate"] = np.log1p(cs[:, t + 1] / (t + 1) * 365.0)   # expected detections / year, history only

    ly = np.zeros((n, T), dtype=np.float32)              # same 15-day window one year earlier (known at t)
    tl = np.nonzero(t >= 365)[0]
    ly[:, tl] = cs[:, tl - 365 + 1 + H] - cs[:, tl - 365 + 1]
    feats["fires_last_year_window"] = np.log1p(ly)

    doy = days.dayofyear.values
    month = days.month.values
    feats["doy_sin"] = np.broadcast_to(np.sin(2 * np.pi * doy / 365.25), (n, T)).astype(np.float32)
    feats["doy_cos"] = np.broadcast_to(np.cos(2 * np.pi * doy / 365.25), (n, T)).astype(np.float32)
    feats["is_kharif"] = np.broadcast_to(np.isin(month, (10, 11)).astype(np.float32), (n, T))

    latlon = np.array([h3.cell_to_latlng(c) for c in cells], dtype=np.float32)
    feats["lat"] = np.broadcast_to(latlon[:, [0]], (n, T))
    feats["lon"] = np.broadcast_to(latlon[:, [1]], (n, T))

    # --- label: any detection in t+1 .. t+H  (NaN where the window is not fully observed) ----
    y = np.full((n, T), np.nan, dtype=np.float32)
    tv = np.arange(0, T - H)
    y[:, tv] = ((cs[:, tv + 1 + H] - cs[:, tv + 1]) > 0).astype(np.float32)

    # --- which days become rows -------------------------------------------------------------
    keep = t >= cfg.burn_in_days
    if last_n_days is not None:
        keep = np.zeros(T, dtype=bool)
        keep[max(0, T - last_n_days):] = True
    elif season_only and cfg.season_months:
        keep &= np.isin(month, cfg.season_months)
    sel = np.nonzero(keep)[0]

    names = feature_names(cfg)
    panel = pd.DataFrame({k: feats[k][:, sel].ravel().astype(np.float32) for k in names})
    panel.insert(0, "date", np.tile(days.values[sel], n))
    panel.insert(1, "h3_id", np.array(cells, dtype=object)[np.repeat(np.arange(n), len(sel))])
    panel["y"] = y[:, sel].ravel()
    return panel, names


def time_split(panel: pd.DataFrame, cfg: Config) -> dict:
    """Forward-chaining split with a purge gap of ``horizon_days`` so label windows never overlap."""
    labelled = panel["y"].notna()
    uniq = np.sort(panel.loc[labelled, "date"].unique())
    train_end = pd.Timestamp(cfg.train_end) if cfg.train_end else pd.Timestamp(uniq[int(0.6 * len(uniq))])
    val_end = pd.Timestamp(cfg.val_end) if cfg.val_end else pd.Timestamp(uniq[int(0.8 * len(uniq))])
    gap = pd.Timedelta(days=cfg.horizon_days)
    d = panel["date"]
    return {
        "train": panel[labelled & (d <= train_end)],
        "val": panel[labelled & (d > train_end + gap) & (d <= val_end)],
        "test": panel[labelled & (d > val_end + gap)],
        "train_end": train_end, "val_end": val_end,
    }
