"""Health and prediction API tests."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_up(self, client: TestClient) -> None:
        """Health returns 200 with modelVersion when model is loaded."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "UP"
        assert "modelVersion" in body

    def test_health_public_with_api_key(self, client: TestClient) -> None:
        """Health endpoint is accessible without auth even when AI_API_KEY is set."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_degraded(self, client_degraded: TestClient) -> None:
        """Health returns 503 when model is not loaded."""
        resp = client_degraded.get("/health")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Model not loaded"


class TestPredictAuth:
    """Tests for POST /predict authentication."""

    def test_predict_401_missing_key(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        """Missing Authorization header should return 401."""
        resp = client.post("/predict", json=valid_payload)
        assert resp.status_code == 401

    def test_predict_401_wrong_key(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        """Wrong API key should return 401."""
        resp = client.post(
            "/predict",
            json=valid_payload,
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_predict_401_non_ascii_key(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        """Non-ASCII key must return 401, not 500 (TypeError from compare_digest).

        httpx TestClient cannot send non-ASCII header values, so we test the
        auth dependency directly to ensure compare_digest with .encode()
        handles non-ASCII without raising TypeError.
        """
        from unittest.mock import MagicMock

        import pytest as _pytest

        from app.auth import verify_api_key
        from app.config import Settings

        creds = MagicMock()
        creds.credentials = "clé-spéciàle"
        settings = Settings(AI_API_KEY="test-secret-key")

        with _pytest.raises(HTTPException) as exc_info:
            verify_api_key(credentials=creds, settings=settings)
        assert exc_info.value.status_code == 401

    def test_predict_200_correct_key(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        """Correct API key should return 200."""
        resp = client.post(
            "/predict",
            json=valid_payload,
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert resp.status_code == 200

    def test_predict_200_no_auth_when_empty_key(
        self, client_no_auth: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        """When AI_API_KEY is empty, /predict works without any header."""
        resp = client_no_auth.post("/predict", json=valid_payload)
        assert resp.status_code == 200


class TestPredictDegraded:
    """Tests for POST /predict when model is not loaded."""

    def test_predict_503_no_model(
        self, client_degraded: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        """Predict returns 503 when model is not loaded."""
        resp = client_degraded.post(
            "/predict",
            json=valid_payload,
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert resp.status_code == 503

    def test_health_503_no_model(self, client_degraded: TestClient) -> None:
        """Health returns 503 when model is not loaded."""
        resp = client_degraded.get("/health")
        assert resp.status_code == 503


class TestPredictValidation:
    """Tests for request validation."""

    def test_extra_fields_rejected(self, client_no_auth: TestClient, valid_payload: dict) -> None:
        resp = client_no_auth.post("/predict", json=valid_payload | {"age": 48, "Class": 1})
        assert resp.status_code == 422
        assert {error["loc"][-1] for error in resp.json()["detail"]} == {"age", "Class"}

    def test_422_invalid_enum(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        """Invalid enum value should return 422."""
        valid_payload["htn"] = "maybe"
        resp = client.post(
            "/predict",
            json=valid_payload,
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert resp.status_code == 422

    def test_422_out_of_range_sg(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        """sg out of range should return 422."""
        valid_payload["sg"] = 2.0
        resp = client.post(
            "/predict",
            json=valid_payload,
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert resp.status_code == 422

    def test_422_missing_field(self, client_no_auth: TestClient, valid_payload: dict) -> None:
        valid_payload.pop("rbc")
        resp = client_no_auth.post("/predict", json=valid_payload)
        assert resp.status_code == 422
        assert resp.json()["detail"][0]["loc"] == ["body", "rbc"]

    def test_accepts_cell_count_aliases(
        self, client_no_auth: TestClient, valid_payload: dict
    ) -> None:
        valid_payload["wbcc"] = valid_payload.pop("wc")
        valid_payload["rbcc"] = valid_payload.pop("rc")
        assert client_no_auth.post("/predict", json=valid_payload).status_code == 200


class TestPredictResponse:
    def test_response_contains_only_label(
        self, client_no_auth: TestClient, valid_payload: dict
    ) -> None:
        resp = client_no_auth.post("/predict", json=valid_payload)
        assert resp.status_code == 200
        assert resp.json() == {"outcome": "NORMAL"}

        resp = client_no_auth.post("/predict", json=valid_payload | {"sc": 2.0})
        assert resp.status_code == 200
        assert resp.json() == {"outcome": "CKD_RISK"}

    def test_x_request_id_header(
        self, client: TestClient, valid_payload: dict[str, Any]
    ) -> None:
        """X-Request-ID header must be present in the response."""
        resp = client.post(
            "/predict",
            json=valid_payload,
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) == 32  # uuid4 hex
