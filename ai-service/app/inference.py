"""Run one-row inference with the model trained on new_model.csv."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.schemas import Outcome, PredictionRequest

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "Bp", "Sg", "Al", "Su", "Rbc", "Bu", "Sc", "Sod", "Pot", "Hemo", "Wbcc", "Rbcc", "Htn"
]
INPUT_FIELDS = [
    "bp", "sg", "al", "su", "rbc", "bu", "sc", "sod", "pot", "hemo", "wc", "rc", "htn"
]


def validate_model(model: Any) -> None:
    """Check the model's classes, feature count and feature order."""
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError(f"Model classes must be [0, 1], got {classes}")

    count = model.n_features_in_
    if count != len(FEATURE_COLUMNS):
        raise ValueError(f"Model must have {len(FEATURE_COLUMNS)} features, got {count}")
    saved_names = getattr(model, "feature_names_in_", None)
    if saved_names is not None and list(saved_names) != FEATURE_COLUMNS:
        raise ValueError(f"Model feature names must match {FEATURE_COLUMNS}")


def load_model(path: str) -> Any:
    """Load the classifier and verify the feature list in its sidecar if present."""
    model = joblib.load(path)
    validate_model(model)

    metrics_path = Path(path).with_suffix(".metrics.json")
    if metrics_path.exists():
        with metrics_path.open() as file:
            saved_features = json.load(file).get("features")
        if saved_features != FEATURE_COLUMNS:
            raise ValueError(f"Feature mismatch in {metrics_path.name}: {saved_features}")
    else:
        logger.warning("No metrics.json sidecar found next to %s", path)
    return model


def predict_outcome(payload: PredictionRequest, model: Any) -> Outcome:
    """Convert the two categorical inputs and return the model's predicted class."""
    features = payload.model_dump()
    features["rbc"] = 1.0 if payload.rbc == "normal" else 0.0
    features["htn"] = 1.0 if payload.htn == "yes" else 0.0
    row = [float(features[field]) for field in INPUT_FIELDS]
    if getattr(model, "feature_names_in_", None) is None:
        input_data = np.array([row])
    else:
        input_data = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    predicted_class = model.predict(input_data)[0]
    if predicted_class == 1:
        return "CKD_RISK"
    if predicted_class == 0:
        return "NORMAL"
    raise ValueError(f"Unsupported predicted class: {predicted_class}")
