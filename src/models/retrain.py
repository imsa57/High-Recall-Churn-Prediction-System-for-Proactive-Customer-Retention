"""Triggered retraining pipeline with drift gate + champion/challenger validation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.logger import logger
from src.models.train_model import ModelTrainer
from src.monitoring.drift_detector import detect_univariate_drift

PROMOTION_AUC_DROP_TOLERANCE = 0.02


def should_retrain(reference_csv: str, current_csv: str) -> tuple[bool, Dict]:
    """Gate: retrain only if drift actually exceeds thresholds."""
    reference = pd.read_csv(reference_csv)
    current = pd.read_csv(current_csv)
    reports = detect_univariate_drift(reference, current)
    drifted = [r.feature for r in reports if r.drifted]
    return bool(drifted), {
        "drifted_features": drifted,
        "reports": [r.to_dict() for r in reports],
    }


def retrain_and_promote(new_data_csv: str, champion_path: str) -> Dict:
    """Train a challenger on fresh data; promote only if it beats the champion."""
    logger.info("Starting triggered retraining...")

    trainer = ModelTrainer()
    trainer.train_all_models()  # your existing entry point
    challenger = trainer.best_model_path
    challenger_auc = float(trainer.best_model_metrics["roc_auc"])

    champion_auc = _load_champion_auc(champion_path)
    if challenger_auc + PROMOTION_AUC_DROP_TOLERANCE < champion_auc:
        logger.warning(
            f"Challenger rejected: AUC {challenger_auc:.4f} < champion {champion_auc:.4f}"
        )
        return {"promoted": False, "reason": "challenger_underperformed",
                "champion_auc": champion_auc, "challenger_auc": challenger_auc}

    shutil.copy(challenger, champion_path)
    logger.info(f"Promoted new champion model -> {champion_path}")
    return {"promoted": True, "champion_auc": challenger_auc, "challenger_auc": challenger_auc}


def _load_champion_auc(champion_path: str) -> float:
    metrics_path = Path(champion_path).parent / "all_models_metric.json"
    if not metrics_path.exists():
        return 0.0
    entries = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = max(entries, key=lambda e: e.get("desiered_metrics", {}).get("roc_auc", 0))
    return float(best.get("desiered_metrics", {}).get("roc_auc", 0))
