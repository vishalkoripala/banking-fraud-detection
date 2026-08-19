"""
tests/test_api.py

Integration tests for the FastAPI service. These load the real trained
artifacts, so run `python train_final_model.py` first if model/ is empty.

Run with: pytest tests/test_api.py
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so model/ resolves

from api import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers the lifespan startup/shutdown
        yield c


@pytest.fixture(scope="module")
def sample_transaction():
    with open("sample_legit_txn.json") as f:
        return json.load(f)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_name" in body
    assert "decision_threshold" in body
    assert "test_metrics" in body
    assert 0.0 <= body["test_metrics"]["pr_auc"] <= 1.0


def test_predict_valid_transaction_returns_full_shape(client, sample_transaction):
    resp = client.post("/predict", json=sample_transaction)
    assert resp.status_code == 200
    body = resp.json()

    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["flagged"], bool)
    assert len(body["top_factors"]) == 5
    for factor in body["top_factors"]:
        assert factor["direction"] in ("increases_risk", "decreases_risk")
    assert body["scoring_time_ms"] > 0


def test_predict_missing_field_returns_422(client, sample_transaction):
    incomplete = dict(sample_transaction)
    del incomplete["V14"]
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_wrong_type_returns_422(client, sample_transaction):
    bad = dict(sample_transaction)
    bad["Amount"] = "not-a-number"
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_negative_amount_returns_422(client, sample_transaction):
    bad = dict(sample_transaction)
    bad["Amount"] = -10.0
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_is_deterministic(client, sample_transaction):
    r1 = client.post("/predict", json=sample_transaction).json()
    r2 = client.post("/predict", json=sample_transaction).json()
    assert r1["fraud_probability"] == r2["fraud_probability"]
