import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "device" in data


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "architecture" in data
    assert "version" in data


def test_predict_endpoint_with_image():
    img = np.ones((32, 128, 3), dtype=np.uint8) * 255
    cv2.putText(img, "TEST", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    _, buffer = cv2.imencode(".png", img)
    image_bytes = io.BytesIO(buffer.tobytes())

    response = client.post(
        "/predict",
        files={"file": ("test_sample.png", image_bytes, "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], (int, float))


def test_index_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Smart OCR" in response.text
