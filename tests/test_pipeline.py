"""Unit tests - run with:  pytest -q   (from the repository root)."""
import h3
import numpy as np
import pandas as pd
import pytest

from stubble_ews.config import Config
from stubble_ews.data import load_firms, make_synthetic_firms
from stubble_ews.evaluate import binary_metrics, pick_thresholds, risk_band
from stubble_ews.features import build_panel, time_split

CFG = Config(h3_res=6)
DASHBOARD_COLUMNS = ["date", "h3_id", "lat", "lon", "h3_parent", "risk_band", "p_fire_15d", "any_exceeds"]


def _firms(rows):
    df = pd.DataFrame(rows, columns=["date", "latitude", "longitude"])
    df["date"] = pd.to_datetime(df["date"])
    df["weight"] = 1.0
    return df


@pytest.fixture
def small_firms():
    rng = np.random.default_rng(0)
    days = pd.date_range("2022-01-01", "2022-06-30")
    rows = []
    for lat, lon in [(31.0, 75.0), (30.4, 75.9), (31.6, 74.8)]:
        for d in rng.choice(days, size=25, replace=False):
            rows.append((d, lat, lon))
    return _firms(rows)


def test_label_matches_brute_force(small_firms):
    panel, _ = build_panel(small_firms, CFG, season_only=False)
    last = small_firms.date.max()
    small_firms = small_firms.assign(h3_id=[h3.latlng_to_cell(a, b, CFG.h3_res)
                                            for a, b in zip(small_firms.latitude, small_firms.longitude)])
    checked = 0
    for cell in small_firms.h3_id.unique():
        fires = small_firms[small_firms.h3_id == cell].date
        sub = panel[panel.h3_id == cell].sample(40, random_state=1)
        for _, r in sub.iterrows():
            d = r["date"]
            if d + pd.Timedelta(days=CFG.horizon_days) > last:
                assert np.isnan(r["y"]), "incomplete label window must be NaN, never 0"
                continue
            expected = float(((fires > d) & (fires <= d + pd.Timedelta(days=CFG.horizon_days))).any())
            assert r["y"] == expected
            checked += 1
    assert checked > 50


def test_features_ignore_the_future(small_firms):
    """Adding a detection on day D must not change any feature on days < D."""
    D = pd.Timestamp("2022-04-10")
    extra = _firms([(D, 31.0, 75.0), (D + pd.Timedelta(days=1), 31.0, 75.0)])
    a, feats = build_panel(small_firms, CFG, season_only=False, date_range=("2022-01-01", "2022-06-30"))
    b, _ = build_panel(pd.concat([small_firms, extra], ignore_index=True), CFG, season_only=False,
                       date_range=("2022-01-01", "2022-06-30"))
    a = a[a.date < D].sort_values(["h3_id", "date"]).reset_index(drop=True)
    b = b[b.date < D].sort_values(["h3_id", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[feats], b[feats])


def test_window_counts(small_firms):
    panel, _ = build_panel(small_firms, CFG, season_only=False)
    cell = h3.latlng_to_cell(31.0, 75.0, CFG.h3_res)
    fires = small_firms[(small_firms.latitude == 31.0)].date
    r = panel[(panel.h3_id == cell)].iloc[60]
    exp = ((fires > r["date"] - pd.Timedelta(days=7)) & (fires <= r["date"])).sum()
    assert r["fires_7d"] == pytest.approx(np.log1p(exp), abs=1e-5)


def test_time_split_is_purged():
    panel = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=1000),
                          "y": np.random.default_rng(0).integers(0, 2, 1000).astype(float)})
    sp = time_split(panel, Config())
    gap = pd.Timedelta(days=15)
    assert sp["train"].date.max() + gap < sp["val"].date.min()
    assert sp["val"].date.max() + gap < sp["test"].date.min()


def test_thresholds_are_ordered_and_bands_work():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.6 * y + rng.normal(0, 0.2, 5000), 0, 1)
    th = pick_thresholds(y, p, Config())
    assert th["high"] > th["medium"] > th["low"] > 0
    bands = risk_band(np.array([0.0, th["low"], th["medium"], th["high"]]), th)
    assert list(bands) == ["None", "Low", "Medium", "High"]


def test_binary_metrics():
    m = binary_metrics(np.array([1, 1, 0, 0, 0]), np.array([1, 0, 1, 0, 0]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["accuracy"] == 0.6


def test_load_firms_aliases_and_quality_filter(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"Lat": [31.0, 31.0, 40.0], "Lon": [75.0, 75.0, 75.0],
                  "Date": ["2024-10-01", "2024-10-02", "2024-10-03"], "Fire Count": [3, 1, 9]}).to_csv(p, index=False)
    df = load_firms(str(p))
    assert len(df) == 2 and df.weight.tolist() == [3, 1]       # 3rd row is outside the Punjab bbox

    q = tmp_path / "viirs.csv"
    pd.DataFrame({"latitude": [31.0, 31.1], "longitude": [75.0, 75.1], "acq_date": ["2024-10-01"] * 2,
                  "confidence": ["l", "h"]}).to_csv(q, index=False)
    assert len(load_firms(str(q))) == 1                         # low-confidence VIIRS dropped


def test_end_to_end_small_run_and_dashboard_schema(tmp_path):
    from stubble_ews.pipeline import run_training
    cfg = Config(h3_res=5, rf_trees=15, xgb_trees=30, train_end="2022-01-31", val_end="2022-12-31")
    firms = make_synthetic_firms(start="2021-01-01", end="2023-12-31", n_hotspots=10)
    res = run_training(firms, cfg, out_dir=str(tmp_path), data_label="SYNTHETIC (unit test)")
    t = res["test"]["at_high_threshold"]
    assert 0 <= t["precision"] <= 1 and 0 <= t["recall"] <= 1
    assert res["test"]["ranking_ensemble"]["pr_auc"] > res["test"]["ranking_ensemble"]["prevalence"]  # beats chance
    alerts = pd.read_csv(tmp_path / "outputs" / "alerts_latest.csv")
    assert list(alerts.columns) == DASHBOARD_COLUMNS
    assert set(alerts.risk_band) <= {"Low", "Medium", "High"}
    assert (tmp_path / "models" / "stubble_ensemble.joblib").exists()
    assert (tmp_path / "reports" / "metrics.json").exists()
