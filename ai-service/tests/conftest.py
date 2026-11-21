"""TestClient and dummy model fixtures."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


class FakeModel:
    classes_ = np.array([0, 1])
    n_features_in_ = 13

    def predict(self, row: np.ndarray) -> np.ndarray:
        return np.array([int(row[0, 6] >= 2.0)])


def _valid_payload() -> dict[str, Any]:
    return {
        "bp": 80,
        "sg": 1.020,
        "al": 0,
        "su": 0,
        "rbc": "normal",
        "bu": 36,
        "sc": 1.1,
        "sod": 138,
        "pot": 4.2,
        "hemo": 14.6,
        "wc": 7800,
        "rc": 5.1,
        "htn": "no",
    }


@pytest.fixture()
def valid_payload() -> dict[str, Any]:
    """The PAT-001 sample row from the spec."""
    return _valid_payload()


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with FakeModel loaded and API key set."""
    monkeypatch.setenv("AI_API_KEY", "test-secret-key")
    get_settings.cache_clear()

    import app.main as main_module

    monkeypatch.setattr(main_module, "load_model", lambda path: FakeModel())

    from app.main import app

    with TestClient(app) as tc:
        yield tc

    get_settings.cache_clear()


@pytest.fixture()
def client_no_auth(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with FakeModel loaded and no API key."""
    monkeypatch.setenv("AI_API_KEY", "")
    get_settings.cache_clear()

    import app.main as main_module

    monkeypatch.setattr(main_module, "load_model", lambda path: FakeModel())

    from app.main import app

    with TestClient(app) as tc:
        yield tc

    get_settings.cache_clear()


@pytest.fixture()
def client_degraded(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient where load_model raises FileNotFoundError (degraded mode)."""
    monkeypatch.setenv("AI_API_KEY", "test-secret-key")
    get_settings.cache_clear()

    import app.main as main_module

    def _fail(path: str) -> None:
        raise FileNotFoundError(f"No model at {path}")

    monkeypatch.setattr(main_module, "load_model", _fail)

    from app.main import app

    with TestClient(app) as tc:
        yield tc

    get_settings.cache_clear()
