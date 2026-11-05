import pytest
from fastapi.testclient import TestClient

from api import main
from src.features.preprocessing import FEATURES
from src.models.train import build_pipeline

PAYLOAD = {
    "age": 40, "job": "admin.", "marital": "married", "education": "high.school",
    "default": "no", "housing": "yes", "loan": "no", "contact": "cellular", "month": "may",
    "day_of_week": "mon", "campaign": 2, "pdays": 999, "previous": 0, "poutcome": "nonexistent",
    "emp_var_rate": 1.1, "cons_price_idx": 93.9, "cons_conf_idx": -36.4, "euribor3m": 4.8,
    "nr_employed": 5191,
}


@pytest.fixture()
def client(split_df, monkeypatch):
    monkeypatch.setattr(main, "reload_model", lambda: None)  # no MLflow in unit tests
    train = split_df[split_df.split == "train"]
    pipe = build_pipeline({"max_depth": 2, "learning_rate": 0.3, "n_estimators": 10})
    pipe.fit(train[FEATURES], train["target"])
    main.state.update(model=pipe, version="1", run_id="abc")
    with TestClient(main.app) as c:
        yield c
    main.state.update(model=None, version=None, run_id=None)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"]


def test_predict(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1) and 0 <= body["probability"] <= 1
    assert body["model_version"] == "1"


def test_predict_validation_error(client):
    bad = {**PAYLOAD, "age": 5}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_without_model(client):
    main.state.update(model=None)
    assert client.post("/predict", json=PAYLOAD).status_code == 503
