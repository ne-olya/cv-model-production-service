from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def png():
    buffer = BytesIO()
    Image.new("RGB", (32, 32), "green").save(buffer, "PNG")
    return buffer.getvalue()


def test_health_and_info():
    assert client.get("/health").json()["status"] == "ok"
    assert "version" in client.get("/model-info").json()


def test_predict_smoke():
    response = client.post("/predict", files={"file": ("test.png", png(), "image/png")})
    assert response.status_code == 200
    assert 0 <= response.json()["confidence"] <= 1


def test_batch():
    files = [("files", ("a.png", png(), "image/png")), ("files", ("b.png", png(), "image/png"))]
    response = client.post("/predict-batch", files=files)
    assert response.status_code == 200 and response.json()["count"] == 2


def test_rejects_bad_format():
    response = client.post("/predict", files={"file": ("x.txt", b"bad", "text/plain")})
    assert response.status_code == 415
