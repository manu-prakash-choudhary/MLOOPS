from fastapi.testclient import TestClient

from app import app

FULL = {
    "age": 63, "sex": "Male", "cp": "asymptomatic", "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": "lv hypertrophy", "thalch": 150, "exang": 0, "oldpeak": 2.3,
    "slope": "downsloping", "ca": 0, "thal": "fixed defect",
}
MINIMAL = {"age": 41, "sex": "Female", "cp": "atypical angina"}


def test_health_and_predict():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}

        r = client.post("/predict", json=FULL)
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["probability"] <= 1
        assert body["prediction"] == int(body["probability"] >= body["threshold"])
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "heart_api_requests_total" in metrics.text

        assert client.post("/predict", json=MINIMAL).status_code == 200


def test_risk_ordering_and_validation():
    with TestClient(app) as client:
        high = client.post("/predict", json={**FULL, "cp": "asymptomatic", "exang": 1}).json()
        low = client.post("/predict", json={**FULL, "cp": "atypical angina", "exang": 0}).json()
        assert high["probability"] > low["probability"]

        assert client.post("/predict", json={**FULL, "sex": "Other"}).status_code == 422
        assert client.post("/predict", json={**FULL, "chol": 0}).status_code == 422
        assert client.post("/predict", json={"age": 50}).status_code == 422
