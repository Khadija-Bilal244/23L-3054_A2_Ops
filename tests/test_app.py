import io

import pytest
from PIL import Image

from app.db import Database
from app.main import create_app


class FakePredictor:
    model_name, model_version, threshold = "fake", "0", 0.5

    def predict_bytes(self, data):
        return {"label": "ai_generated", "confidence": 0.9, "prob_ai": 0.9, "threshold": 0.5,
                "model_name": "fake", "model_version": "0", "latency_ms": 1.0}


@pytest.fixture()
def client():
    app = create_app(FakePredictor(), Database("sqlite://"))
    app.config["TESTING"] = False  # exercise the real error handlers
    return app.test_client()


def png_bytes():
    b = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(b, "PNG")
    return b.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_predict_and_history(client):
    r = client.post("/predict", data={"image": (io.BytesIO(png_bytes()), "a b.png")})
    assert r.status_code == 200 and b"Likely AI-generated" in r.data
    h = client.get("/history")
    assert b"a_b.png" in h.data  # sanitized file name stored


def test_wrong_type(client):
    r = client.post("/predict", data={"image": (io.BytesIO(b"hello"), "x.txt")})
    assert r.status_code == 415


def test_fake_image(client):
    r = client.post("/predict", data={"image": (io.BytesIO(b"not an image"), "x.png")})
    assert r.status_code == 422


def test_too_large(client, monkeypatch):
    from app.settings import Settings
    monkeypatch.setattr(Settings, "MAX_UPLOAD_MB", 0)
    r = client.post("/predict", data={"image": (io.BytesIO(png_bytes()), "x.png")})
    assert r.status_code == 413


def test_404(client):
    assert client.get("/nope").status_code == 404
