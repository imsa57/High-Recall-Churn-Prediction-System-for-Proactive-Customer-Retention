"""Drift detection utilities for the churn prediction pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import roc_auc_score

from src.logger import logger

# Mirrors the feature split used in src/features/build_features.py
CONTINUOUS_FEATURES = ["MonthlyCharges", "tenure"]
CATEGORICAL_FEATURES = [
    "SeniorCitizen", "Partner", "Dependents", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "StreamingMovies", "PaperlessBilling", "PaymentMethod", "Contract",
]
DRIFT_THRESHOLDS = {"ks": 0.1, "psi": 0.25, "chi2_pvalue": 0.05}


@dataclass
class DriftReport:
    feature: str
    drift_type: str
    statistic: float
    threshold: float
    drifted: bool
    detail: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    """PSI between reference and current distributions."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty:
        return float("nan")

    # Quantile edges from the reference distribution
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts = np.histogram(ref, bins=edges)[0]
    cur_counts = np.histogram(cur, bins=edges)[0]

    # Laplace smoothing avoids div-by-zero on empty bins
    eps = 1e-6
    ref_pct = (ref_counts / max(ref_counts.sum(), 1)) + eps
    cur_pct = (cur_counts / max(cur_counts.sum(), 1)) + eps

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_univariate_drift(
    reference: pd.DataFrame, current: pd.DataFrame
) -> List[DriftReport]:
    """Per-feature drift via KS (continuous) and PSI (all)."""
    reports: List[DriftReport] = []

    for feature in CONTINUOUS_FEATURES:
        if feature not in reference or feature not in current:
            continue

        ref = pd.to_numeric(reference[feature], errors="coerce").dropna()
        cur = pd.to_numeric(current[feature], errors="coerce").dropna()
        if len(ref) < 30 or len(cur) < 30:
            continue

        ks_stat, p_value = ks_2samp(ref, cur)
        psi = population_stability_index(ref, cur)
        drifted = (ks_stat > DRIFT_THRESHOLDS["ks"]) or (
            not np.isnan(psi) and psi > DRIFT_THRESHOLDS["psi"]
        )
        reports.append(
            DriftReport(
                feature=feature,
                drift_type="continuous",
                statistic=round(float(ks_stat), 4),
                threshold=DRIFT_THRESHOLDS["ks"],
                drifted=bool(drifted),
                detail=f"ks_pvalue={p_value:.2e}, psi={psi:.4f}",
            )
        )

    for feature in CATEGORICAL_FEATURES:
        if feature not in reference or feature not in current:
            continue

        ref = reference[feature].astype(str).value_counts(normalize=True)
        cur = current[feature].astype(str).value_counts(normalize=True)
        if len(ref) < 30 or len(cur) < 30:
            continue

        # PSI works for categoricals too
        psi = population_stability_index(
            reference[feature].astype(str), current[feature].astype(str)
        )

        # Chi-square on the aligned contingency table
        categories = ref.index.union(cur.index)
        ref_aligned = ref.reindex(categories).fillna(0.0)
        cur_aligned = cur.reindex(categories).fillna(0.0)
        table = np.vstack([ref_aligned.to_numpy(), cur_aligned.to_numpy()])
        try:
            chi2_stat, p_value, _, _ = chi2_contingency(table)
        except ValueError:
            chi2_stat, p_value = float("nan"), 1.0

        # New / unseen categories are a strong drift signal
        unseen = sorted(set(cur.index) - set(ref.index))
        drifted = (
            p_value < DRIFT_THRESHOLDS["chi2_pvalue"]
            or (not np.isnan(psi) and psi > DRIFT_THRESHOLDS["psi"])
            or len(unseen) > 0
        )
        reports.append(
            DriftReport(
                feature=feature,
                drift_type="categorical",
                statistic=round(float(chi2_stat), 4) if not np.isnan(chi2_stat) else float("nan"),
                threshold=DRIFT_THRESHOLDS["chi2_pvalue"],
                drifted=bool(drifted),
                detail=f"p_value={p_value:.2e}, psi={psi:.4f}, unseen={unseen}",
            )
        )
    return reports


def detect_multivariate_drift(
    reference: pd.DataFrame, current: pd.DataFrame, features: List[str], seed: int = 42
) -> Dict:
    """Train a discriminator to separate reference from current.

    AUC ~0.5 => no joint drift. AUC > 0.7 => the samples come from
    genuinely different distributions (covariate shift).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    common = [f for f in features if f in reference and f in current]
    if len(common) < 2:
        return {"auc": float("nan"), "drifted": False}

    ref_x = reference[common].copy()
    cur_x = current[common].copy()
    ref_x["_is_current"] = 0
    cur_x["_is_current"] = 1
    combined = pd.concat([ref_x, cur_x], ignore_index=True)

    y = combined.pop("_is_current")
    # Simple imputation so NaNs don't break the discriminator
    combined = combined.fillna(combined.mode().iloc[0])
    combined = combined.astype({c: "category" for c in combined.select_dtypes("object").columns})
    combined = combined.apply(lambda s: s.cat.codes if s.dtype.name == "category" else s)

    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    auc = cross_val_score(clf, combined, y, cv=3, scoring="roc_auc").mean()
    return {"auc": round(float(auc), 4), "drifted": bool(auc > 0.7)}


def detect_concept_drift(
    y_true: np.ndarray, y_pred_proba: np.ndarray, baseline_roc: float, tolerance: float = 0.05
) -> Dict:
    """Compare live performance against the training-time baseline."""
    from sklearn.metrics import roc_auc_score

    if len(y_true) < 50:
        return {"drifted": False, "note": "not enough labels yet"}
    live_auc = roc_auc_score(y_true, y_pred_proba)
    return {
        "live_roc_auc": round(float(live_auc), 4),
        "baseline_roc_auc": round(float(baseline_roc), 4),
        "drifted": bool(live_auc < baseline_roc - tolerance),
    }
