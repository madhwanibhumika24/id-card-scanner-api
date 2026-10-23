
from io import BytesIO

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def create_test_image():
    """Create a simple synthetic image for API testing."""
    image = np.full((360, 600, 3), 245, dtype=np.uint8)

    cv2.rectangle(image, (40, 40), (560, 320), (0, 0, 0), 3)
    cv2.putText(
        image,
        "STUDENT ID",
        (170, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        image,
        "Name: Test Student",
        (100, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )

    success, encoded = cv2.imencode(".jpg", image)
    assert success

    return encoded.tobytes()


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"].lower()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_reject_unsupported_file():
    response = client.post(
        "/api/v1/process-id",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_reject_corrupted_image():
    response = client.post(
        "/api/v1/process-id",
        files={"file": ("broken.jpg", b"not a valid image", "image/jpeg")},
    )
    assert response.status_code == 400


def test_missing_record():
    response = client.get(
        "/api/v1/records/00000000-0000-0000-0000-000000000000"
    )
    assert response.status_code == 404


def test_process_and_retrieve_image():
    response = client.post(
        "/api/v1/process-id",
        files={
            "file": (
                "test-card.jpg",
                BytesIO(create_test_image()),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()
    assert result["id"]
    assert result["processing_status"] == "preprocessed"
    assert "full_text" in result["ocr"]
    assert "full_name" in result["extracted_fields"]

    record_response = client.get(
        f"/api/v1/records/{result['id']}"
    )

    assert record_response.status_code == 200
    assert record_response.json()["id"] == result["id"]
