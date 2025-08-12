from fastapi.testclient import TestClient

from api.server import app


def test_health_and_predict_smoke(monkeypatch):
    client = TestClient(app)
    # Health
    r = client.get("/health")
    assert r.status_code == 200 and r.json().get("status") == "ok"

    # Predict offline to avoid network/API keys
    payload = {"ticker": "AAPL", "timeframe": "1d", "low_api_mode": True}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.json().get("status") == "success"


