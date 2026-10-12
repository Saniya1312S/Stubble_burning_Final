"""End-to-end orchestration: panel -> split -> train -> threshold -> evaluate -> save -> export alerts."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import h3
import joblib
import numpy as np
import pandas as pd

from . import plots
from .config import Config
from .evaluate import binary_metrics, pick_thresholds, ranking_metrics, risk_band
from .features import build_panel, time_split
from .models import fit_ensemble


def _describe_split(name: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0}
    return {"rows": int(len(df)), "from": str(df.date.min().date()), "to": str(df.date.max().date()),
            "positive_rate": float(df.y.mean())}


def export_alerts(bundle: dict, firms: pd.DataFrame, cfg: Config, path: str) -> pd.DataFrame:
    """Score the most recent ``cfg.alert_days`` days and write a CSV in the dashboard schema:
    ``date, h3_id, lat, lon, h3_parent, risk_band, p_fire_15d, any_exceeds``."""
    panel, feats = build_panel(firms, cfg, last_n_days=cfg.alert_days)
    p = bundle["ensemble"].predict_proba(panel[bundle["features"]].values)
    th = bundle["thresholds"]
    parent_res = max(cfg.h3_res - 2, 0)
    out = pd.DataFrame({
        "date": panel["date"].dt.strftime("%Y-%m-%d"),
        "h3_id": panel["h3_id"],
        "lat": panel["lat"].astype(float).round(6),
        "lon": panel["lon"].astype(float).round(6),
        "h3_parent": [h3.cell_to_parent(c, parent_res) for c in panel["h3_id"]],
        "risk_band": risk_band(p, th),
        "p_fire_15d": p,
        "any_exceeds": (p >= th["high"]).astype(int),
    })
    out = out[out.risk_band != "None"].sort_values(["date", "p_fire_15d"], ascending=[True, False])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)
    return out


def run_training(firms: pd.DataFrame, cfg: Config, *, out_dir: str = ".", data_label: str = "FIRMS") -> dict:
    fig_dir = os.path.join(out_dir, "reports", "figures")
    plots.WATERMARK = "SYNTHETIC DEMO DATA" if data_label.upper().startswith("SYNTHETIC") else ""
    panel, feats = build_panel(firms, cfg)
    sp = time_split(panel, cfg)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    summary = {"train": _describe_split("train", tr), "val": _describe_split("val", va),
               "test": _describe_split("test", te)}
    print("\n=== Temporal split (purged by %d days) ===" % cfg.horizon_days)
    for k, v in summary.items():
        print(f"  {k:5s} {v}")
    for k, d in (("train", tr), ("val", va), ("test", te)):
        if d.empty or d.y.sum() == 0 or d.y.sum() == len(d):
            raise ValueError(f"The {k} split has no positives/negatives (rows={len(d)}). "
                             "Provide more data or set --train-end / --val-end explicitly.")

    Xtr, ytr = tr[feats].values, tr.y.values
    Xva, yva = va[feats].values, va.y.values
    Xte, yte = te[feats].values, te.y.values

    print("\nTraining ensemble (LogReg + RandomForest + XGBoost) ...")
    ens = fit_ensemble(Xtr, ytr, Xva, yva, cfg)

    p_val, p_te = ens.predict_proba(Xva), ens.predict_proba(Xte)
    th = pick_thresholds(yva, p_val, cfg)
    yhat_hi = p_te >= th["high"]
    yhat_med = p_te >= th["medium"]
    persistence = te["fires_14d"].values > 0            # naive rule: any detection in the last 14 days

    members_te = ens.member_scores(Xte)
    results = {
        "data": {"label": data_label, "n_detections": int(len(firms)),
                 "first_date": str(firms.date.min().date()), "last_date": str(firms.date.max().date())},
        "config": asdict(cfg), "split": summary, "thresholds": th,
        "test": {
            "ranking_ensemble": ranking_metrics(yte, p_te),
            "ranking_members": {k: ranking_metrics(yte, s) for k, s in members_te.items()},
            "at_high_threshold": binary_metrics(yte, yhat_hi, cfg.beta),
            "at_medium_threshold": binary_metrics(yte, yhat_med, cfg.beta),
            "baseline_persistence_14d": binary_metrics(yte, persistence, cfg.beta),
        },
    }

    # ---- figures ----------------------------------------------------------------------
    b = results["test"]["baseline_persistence_14d"]
    plots.pr_curves(yte, {"ensemble": p_te, **members_te}, (b["precision"], b["recall"]), fig_dir)
    plots.roc_curves(yte, {"ensemble": p_te, **members_te}, fig_dir)
    plots.threshold_tradeoff(yva, p_val, th, cfg.beta, fig_dir)
    plots.confusion(results["test"]["at_high_threshold"], fig_dir)
    plots.daily_alerts(te.date.values, yte, yhat_hi, fig_dir)
    imps = {k: m.feature_importances_ for k, m in ens.members.items() if hasattr(m, "feature_importances_")}
    if imps:
        plots.feature_importance(feats, imps, fig_dir)

    # ---- persist ------------------------------------------------------------------------
    bundle = {"ensemble": ens, "features": feats, "thresholds": th, "cfg": cfg}
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    joblib.dump(bundle, os.path.join(out_dir, "models", "stubble_ensemble.joblib"))
    with open(os.path.join(out_dir, "reports", "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    alerts = export_alerts(bundle, firms, cfg, os.path.join(out_dir, "outputs", "alerts_latest.csv"))
    results["alerts_exported"] = int(len(alerts))

    _print_report(results)
    return results


def _print_report(r: dict) -> None:
    t = r["test"]
    print("\n=== Test-set results (held-out future period) ===")
    print(f"  prevalence            : {t['ranking_ensemble']['prevalence']:.3f}")
    print(f"  ensemble PR-AUC       : {t['ranking_ensemble']['pr_auc']:.3f}   ROC-AUC: {t['ranking_ensemble']['roc_auc']:.3f}")
    for name in ("at_high_threshold", "at_medium_threshold", "baseline_persistence_14d"):
        m = t[name]
        print(f"  {name:26s} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
              f"acc={m['accuracy']:.3f}  alert_rate={m['alert_rate']:.3f}")
    print(f"  thresholds            : {r['thresholds']}")
    print(f"  alerts exported       : {r.get('alerts_exported', '-')} rows")
