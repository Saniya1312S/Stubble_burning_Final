"""Command-line interface:  python -m stubble_ews <command>"""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib

from .config import Config
from .data import load_firms, make_synthetic_firms
from .eda_alerts import analyse
from .pipeline import export_alerts, run_training


def _cfg_from(args) -> Config:
    cfg = Config(h3_res=args.h3_res, train_end=args.train_end, val_end=args.val_end)
    if getattr(args, "all_year", False):
        cfg.season_months = None
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stubble_ews", description="Stubble-burning early-warning pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--h3-res", type=int, default=6, help="H3 resolution (default 6; sample CSV uses 7)")
        p.add_argument("--train-end", default=None, help="last training day, e.g. 2023-05-31")
        p.add_argument("--val-end", default=None, help="last validation day, e.g. 2024-02-29")
        p.add_argument("--all-year", action="store_true", help="use all months (default: Sep-Dec stubble season)")
        p.add_argument("--out", default=".", help="output root (models/, reports/, outputs/)")

    p = sub.add_parser("demo", help="run the full pipeline on SYNTHETIC data (smoke test / demo only)")
    common(p)
    p = sub.add_parser("train", help="train on a real NASA FIRMS CSV")
    p.add_argument("--firms", required=True, help="FIRMS CSV (VIIRS/MODIS) or Lat,Lon,Date[,Fire Count] table")
    common(p)
    p = sub.add_parser("predict", help="score recent days with a saved model and export dashboard-ready alerts")
    p.add_argument("--firms", required=True)
    p.add_argument("--model", default="models/stubble_ensemble.joblib")
    p.add_argument("--out-csv", default="outputs/alerts_latest.csv")
    p = sub.add_parser("eda-alerts", help="analyse the alert CSV used by the dashboard")
    p.add_argument("--csv", default="dashboard/sample_data/stubble_alerts_latest.csv")
    p.add_argument("--out", default=".")

    a = ap.parse_args(argv)
    if a.cmd == "demo":
        cfg = _cfg_from(a)
        cfg.train_end = cfg.train_end or "2023-01-31"
        cfg.val_end = cfg.val_end or "2023-12-31"
        firms = make_synthetic_firms()
        print(f"[demo] SYNTHETIC data: {len(firms):,} detections - results are NOT real findings.")
        run_training(firms, cfg, out_dir=a.out, data_label="SYNTHETIC (demo only)")
    elif a.cmd == "train":
        cfg = _cfg_from(a)
        firms = load_firms(a.firms, cfg)
        print(f"Loaded {len(firms):,} detections {firms.date.min().date()} -> {firms.date.max().date()}")
        run_training(firms, cfg, out_dir=a.out, data_label=os.path.basename(a.firms))
    elif a.cmd == "predict":
        bundle = joblib.load(a.model)
        firms = load_firms(a.firms, bundle["cfg"])
        out = export_alerts(bundle, firms, bundle["cfg"], a.out_csv)
        print(f"Wrote {len(out):,} alert rows -> {a.out_csv}")
    elif a.cmd == "eda-alerts":
        print(json.dumps(analyse(a.csv, a.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
