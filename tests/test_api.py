from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app
from tests.test_inference import VALID_FEATURES


def test_health_registration_prediction_and_invalid_input() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["database"] == "reachable"
        assert health.json()["feature_count"] == 14

        registered = client.post(
            "/register",
            data={
                "hotel_name": "Nile View",
                "location": "Cairo",
                "email": "owner@example.com",
                "password": "password123",
            },
            follow_redirects=False,
        )
        assert registered.status_code == 302
        assert registered.headers["location"] == "/predict"

        predicted = client.post(
            "/api/v1/predictions",
            json={"features": VALID_FEATURES, "actual_outcome": 0},
        )
        assert predicted.status_code == 200
        body = predicted.json()
        assert body["predicted_outcome"] in {0, 1}
        assert 0 <= body["cancellation_probability"] <= 1
        assert body["model_version"].startswith("xgb-")
        assert body["latency_ms"] >= 0

        invalid = client.post(
            "/api/v1/predictions",
            json={"features": {**VALID_FEATURES, "lead_time": 9999}},
        )
        assert invalid.status_code == 422


def test_prediction_api_requires_login() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/predictions", json={"features": VALID_FEATURES})
        assert response.status_code == 401
