"""Loading fire-detection data (NASA FIRMS) and a synthetic generator for demos/tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PUNJAB_BBOX, Config

_ALIASES = {
    "latitude": ["latitude", "lat", "Latitude", "LATITUDE"],
    "longitude": ["longitude", "lon", "lng", "long", "Longitude", "LONGITUDE"],
    "date": ["acq_date", "date", "Date", "DATE", "acq_datetime"],
    "fire_count": ["fire_count", "Fire Count", "fire_counts", "count"],
}


def _pick(df: pd.DataFrame, key: str, required: bool = True):
    lower = {str(c).strip().lower(): c for c in df.columns}      # case/whitespace-insensitive match
    for c in _ALIASES[key]:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"Could not find a '{key}' column. Tried {_ALIASES[key]}. Found: {list(df.columns)}")
    return None


def load_firms(path: str, cfg: Config | None = None, bbox: dict | None = None) -> pd.DataFrame:
    """Read a NASA FIRMS CSV (VIIRS or MODIS) -> tidy frame [date, latitude, longitude, weight].

    Accepts the official FIRMS export (``latitude, longitude, acq_date, confidence ...``) and
    simple ``Lat, Lon, Date, Fire Count`` tables like the ones used in the internship Colab.
    """
    cfg = cfg or Config()
    bbox = bbox or PUNJAB_BBOX
    raw = pd.read_csv(path)
    lat, lon, dt = _pick(raw, "latitude"), _pick(raw, "longitude"), _pick(raw, "date")
    cnt = _pick(raw, "fire_count", required=False)

    df = pd.DataFrame({
        "date": pd.to_datetime(raw[dt], errors="coerce").dt.normalize(),
        "latitude": pd.to_numeric(raw[lat], errors="coerce"),
        "longitude": pd.to_numeric(raw[lon], errors="coerce"),
        "weight": pd.to_numeric(raw[cnt], errors="coerce") if cnt else 1.0,
    })

    if "confidence" in raw.columns:                       # quality filtering (FIRMS only)
        conf = raw["confidence"]
        if pd.api.types.is_numeric_dtype(conf):           # MODIS: 0-100
            df = df[(conf >= cfg.min_modis_confidence).values]
        elif cfg.drop_low_confidence_viirs:               # VIIRS: l / n / h  (works on pandas 2 and 3)
            keep = ~conf.astype(str).str.strip().str.lower().isin(["l", "low"])
            df = df[keep.values]

    df = df.dropna(subset=["date", "latitude", "longitude"])
    df["weight"] = df["weight"].fillna(1.0)
    inside = (df.latitude.between(bbox["lat_min"], bbox["lat_max"])
              & df.longitude.between(bbox["lon_min"], bbox["lon_max"]))
    return df[inside].sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Synthetic data  (FOR DEMOS AND UNIT TESTS ONLY - never report its metrics as findings)
# --------------------------------------------------------------------------------------
def make_synthetic_firms(start="2021-01-01", end="2024-12-31", n_hotspots=14, seed=7,
                         bbox: dict | None = None) -> pd.DataFrame:
    """Simulate Punjab-like fire detections: a big Oct-Nov (Kharif / paddy) peak, a smaller
    Apr-May (Rabi / wheat) peak, spatial hotspots and multi-day bursts.

    This only exists so the pipeline can be run end-to-end without a NASA FIRMS download.
    """
    rng = np.random.default_rng(seed)
    bbox = bbox or PUNJAB_BBOX
    days = pd.date_range(start, end, freq="D")
    doy = days.dayofyear.values
    season = (1.0 * np.exp(-0.5 * ((doy - 300) / 14.0) ** 2)      # ~27 Oct
              + 0.25 * np.exp(-0.5 * ((doy - 110) / 10.0) ** 2)   # ~20 Apr
              + 0.01)
    year_factor = {y: rng.uniform(0.7, 1.3) for y in sorted(set(days.year))}
    yf = np.array([year_factor[y] for y in days.year])

    # spatial hotspots (cluster centres) with different intensities
    centres = np.column_stack([rng.uniform(bbox["lat_min"] + .3, bbox["lat_max"] - .3, n_hotspots),
                               rng.uniform(bbox["lon_min"] + .3, bbox["lon_max"] - .3, n_hotspots)])
    weights = rng.lognormal(0.0, 0.6, n_hotspots)

    rows = []
    for k, (c, w) in enumerate(zip(centres, weights)):
        burst = np.zeros(len(days))                                # AR(1) log-burst -> multi-day persistence
        for i in range(1, len(days)):
            burst[i] = 0.85 * burst[i - 1] + rng.normal(0, 0.6)
        lam = 6.0 * w * season * yf * np.exp(burst)
        n = rng.poisson(np.clip(lam, 0, 150))
        for d_idx in np.nonzero(n)[0]:
            m = n[d_idx]
            rows.append(pd.DataFrame({
                "date": days[d_idx],
                "latitude": np.clip(c[0] + rng.normal(0, 0.18, m), bbox["lat_min"], bbox["lat_max"]),
                "longitude": np.clip(c[1] + rng.normal(0, 0.18, m), bbox["lon_min"], bbox["lon_max"]),
                "weight": 1.0,
            }))
    return pd.concat(rows, ignore_index=True).sort_values("date").reset_index(drop=True)
