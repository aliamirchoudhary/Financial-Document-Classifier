"""API smoke tests. Run from repo root with the env vars set:
    $env:MODEL_DIR=... ; $env:AUTH_TOKEN=secret
    python -m pytest tests/test_api.py
"""
import os
import fastapi.testclient
from app.main import app

client = fastapi.testclient.TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    assert r.json()["num_classes"] == 17


def test_classes_requires_token():
    r = client.get("/classes")
    assert r.status_code == 401


def test_classes_with_token():
    r = client.get("/classes", headers={"Authorization": "Bearer " + os.environ.get("AUTH_TOKEN", "")})
    assert r.status_code == 200
    assert len(r.json()["classes"]) == 17


def test_predict_text():
    r = client.post("/predict_text",
                    json={"text": "Trial balance report for December 2024, account debit credit"},
                    headers={"Authorization": "Bearer " + os.environ.get("AUTH_TOKEN", "")})
    assert r.status_code == 200
    assert "predicted_class" in r.json()
    assert r.json()["num_classes"] == 17


def test_predict_requires_token():
    r = client.post("/predict", files={"file": ("x.txt", b"hello")})
    assert r.status_code == 401