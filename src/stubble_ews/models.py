"""Model zoo: Logistic Regression + Random Forest + XGBoost, combined by soft voting."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config

try:                                                    # XGBoost is optional (falls back to HistGradientBoosting)
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:                                       # pragma: no cover
    HAS_XGB = False


class SoftVoteEnsemble:
    """Average of member probabilities, optionally followed by an isotonic calibrator."""

    def __init__(self, members: Dict[str, object]):
        self.members = members
        self.calibrator: IsotonicRegression | None = None

    def member_scores(self, X) -> Dict[str, np.ndarray]:
        return {k: m.predict_proba(X)[:, 1] for k, m in self.members.items()}

    def raw_score(self, X) -> np.ndarray:
        return np.mean(list(self.member_scores(X).values()), axis=0)

    def fit_calibrator(self, X_val, y_val):
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(self.raw_score(X_val), y_val)
        return self

    def predict_proba(self, X) -> np.ndarray:
        s = self.raw_score(X)
        return self.calibrator.predict(s) if self.calibrator is not None else s


def fit_ensemble(X_tr, y_tr, X_val, y_val, cfg: Config) -> SoftVoteEnsemble:
    pos = max(float(y_tr.sum()), 1.0)
    neg = float(len(y_tr) - y_tr.sum())
    members: Dict[str, object] = {}

    members["logreg"] = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", C=1.0),
    ).fit(X_tr, y_tr)

    members["random_forest"] = RandomForestClassifier(
        n_estimators=cfg.rf_trees, max_depth=14, min_samples_leaf=20,
        max_samples=cfg.rf_max_samples, class_weight="balanced_subsample",
        n_jobs=-1, random_state=cfg.random_state,
    ).fit(X_tr, y_tr)

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=cfg.xgb_trees, max_depth=6, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", scale_pos_weight=neg / pos,
            eval_metric="aucpr", early_stopping_rounds=30, n_jobs=-1,
            random_state=cfg.random_state, verbosity=0,
        )
        xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        members["xgboost"] = xgb
    else:                                               # pragma: no cover
        members["hist_gb"] = HistGradientBoostingClassifier(
            class_weight="balanced", random_state=cfg.random_state).fit(X_tr, y_tr)

    ens = SoftVoteEnsemble(members)
    ens.fit_calibrator(X_val, y_val)
    return ens
