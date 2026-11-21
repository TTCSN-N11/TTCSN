"""Check the training column order, category encoding and class mapping."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.inference import FEATURE_COLUMNS, INPUT_FIELDS, predict_outcome, validate_model
from app.schemas import PredictionRequest


class Classifier:
    classes_ = np.array([0, 1])
    n_features_in_ = 13
    feature_names_in_ = np.array(FEATURE_COLUMNS)

    def __init__(self, label: int) -> None:
        self.label = label

    def predict(self, row: pd.DataFrame) -> np.ndarray:
        assert list(row.columns) == FEATURE_COLUMNS
        np.testing.assert_allclose(
            row.to_numpy(), [[80, 1.020, 0, 0, 1, 36, 1.1, 138, 4.2, 14.6, 7800, 5.1, 0]]
        )
        return np.array([self.label])


@pytest.mark.parametrize(("label", "outcome"), [(0, "NORMAL"), (1, "CKD_RISK")])
def test_maps_estimator_label(label, outcome, valid_payload):
    assert predict_outcome(PredictionRequest(**valid_payload), Classifier(label)) == outcome


def test_model_feature_order_matches_training_header():
    path = Path(__file__).resolve().parents[1] / "training" / "data" / "new_model.csv"
    with path.open() as file:
        assert next(csv.reader(file)) == [*FEATURE_COLUMNS, "Class"]
    assert list(PredictionRequest.model_fields) == INPUT_FIELDS
    assert all(field.is_required() for field in PredictionRequest.model_fields.values())


def test_rejects_another_feature_count():
    model = Classifier(0)
    model.n_features_in_ = 24
    with pytest.raises(ValueError, match="Model must have 13 features"):
        validate_model(model)


def test_rejects_wrong_feature_order():
    model = Classifier(0)
    model.feature_names_in_ = model.feature_names_in_[::-1]
    with pytest.raises(ValueError, match="Model feature names"):
        validate_model(model)


def test_rejects_wrong_classes():
    model = Classifier(0)
    model.classes_ = np.array([1, 0])
    with pytest.raises(ValueError, match="Model classes must be"):
        validate_model(model)
