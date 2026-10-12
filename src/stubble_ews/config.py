"""Central configuration for the pipeline (single source of truth)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# Rough bounding box of Punjab (India). Used to build the hexagon grid.
# NOTE: a bbox also covers slivers of Haryana / Himachal / Pakistan; replace with a
# proper Punjab boundary polygon (e.g. FAO GAUL level-1) for publication-grade work.
PUNJAB_BBOX = dict(lat_min=29.5, lat_max=32.6, lon_min=73.8, lon_max=77.0)


@dataclass
class Config:
    # ---- spatial / temporal definition of the problem ----
    h3_res: int = 6                       # H3 resolution (6 ~ 36 km2 hexagons; 7 matches the sample CSV)
    horizon_days: int = 15                # forecast window: fire in days t+1 .. t+15
    windows: Tuple[int, ...] = (1, 3, 7, 14, 30, 90)    # look-back windows (days) for fire counts
    burn_in_days: int = 30                # first N days of the series are dropped (truncated look-back)
    season_months: Optional[Tuple[int, ...]] = (9, 10, 11, 12)  # rows kept for train/eval; None = all year

    # ---- data quality ----
    min_modis_confidence: int = 30        # drop MODIS detections below this (0-100)
    drop_low_confidence_viirs: bool = True  # drop VIIRS confidence == 'l'

    # ---- temporal split (forward-chaining, purged by horizon_days) ----
    train_end: Optional[str] = None       # e.g. "2023-05-31"; None -> 60 % quantile of dates
    val_end: Optional[str] = None         # e.g. "2024-02-29"; None -> 80 % quantile of dates

    # ---- modelling ----
    beta: float = 2.0                     # F-beta used to pick the operating threshold (2 => favour recall)
    medium_recall_target: float = 0.90    # 'Medium' band = highest threshold that still reaches this recall (val)
    rf_trees: int = 150
    rf_max_samples: float = 0.15
    xgb_trees: int = 300
    random_state: int = 42

    # ---- alert export ----
    alert_days: int = 30                  # export alerts for the most recent N days of the input data
    extra: dict = field(default_factory=dict)
