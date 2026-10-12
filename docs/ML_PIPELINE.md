# ML pipeline design (`src/stubble_ews/`)

**Task.** For every H3 hexagon and day *t*: will a satellite fire detection occur in days *t+1 ... t+15*? (binary)

| Step | File | What happens |
|---|---|---|
| 1 Load | `data.py` | FIRMS CSV -> clean table; bbox + confidence filters |
| 2 Grid & features | `features.py` | hex grid over Punjab; daily counts per cell; look-back windows (1, 3, 7, 14, 30, 90 d); neighbour-ring counts; days since last fire; expected yearly rate; same-window-last-year; seasonality; lat/lon. Counts are `log1p`-scaled |
| 3 Label | `features.py` | `y = any detection in t+1..t+15`; **NaN** when the window is not fully observed |
| 4 Split | `features.py` | forward-chaining train / val / test with a 15-day purge gap |
| 5 Train | `models.py` | LogReg + Random Forest + XGBoost (class-weighted), soft vote, isotonic calibration on val |
| 6 Thresholds | `evaluate.py` | High = F2-optimal on **val**; Medium <= 0.75 x High at 90 % recall; Low = Medium / 2 |
| 7 Evaluate | `evaluate.py`, `plots.py` | precision, recall, F1, F2, PR-AUC, ROC-AUC, Brier, confusion matrix, **persistence baseline** |
| 8 Export | `pipeline.py` | alerts CSV in the dashboard's schema |

## Leakage controls (tested in `tests/test_pipeline.py`)
* features at day *t* use data <= *t* only (perturbation test: adding future fires changes no earlier feature);
* label alignment verified against a brute-force computation;
* purge gap between splits so label windows never overlap;
* thresholds chosen on validation, never on test.

## Why not just report accuracy?
When ~0.4 % (alerts CSV) to ~13 % (demo) of cell-days are positive, "always predict no fire" already scores
99.6 % / 87 % accuracy. We report PR-AUC, recall and precision next to a persistence baseline instead.

## Known limitations
* The grid is a rectangle around Punjab, not the official boundary.
* FIRMS detections are a proxy for burning, not a measured burned area.
* Scores are calibrated on one validation season; re-calibrate each year.
* The synthetic generator (`data.py`) exists only for demos and unit tests.
