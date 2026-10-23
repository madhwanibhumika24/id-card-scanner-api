
from io import BytesIO

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def create_test_image():
    """Create a simple synthetic image for upload tests."""
    image = np.full((360, 600, 3), 240, dtype=np.uint8)

    cv2.rectangle(
        image,
        (30, 30),
        (570, 330),
        (20, 20, 20),
        4,
    )

    cv2.putText(
        image,
        "SAMPLE ID CARD",
        (70, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    success, encoded = cv2.imencode(".jpg", image)
    assert success

    return encoded.tobytes()


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "ID Card Scanner API is running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_reject_unsupported_file():
    response = client.post(
        "/api/v1/process-id",
        files={
            "file": (
                "sample.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_reject_unreadable_image():
    response = client.post(
        "/api/v1/process-id",
        files={
            "file": (
                "fake.jpg",
                b"not really an image",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400


def test_record_not_found():
    response = client.get(
        "/api/v1/records/nonexistent-test-record"
    )

    assert response.status_code == 404


def test_process_image():
    image_bytes = create_test_image()

    response = client.post(
        "/api/v1/process-id",
        files={
            "file": (
                "sample.jpg",
                BytesIO(image_bytes),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["id"]
    assert result["metadata"]["original_width"] == 600
    assert result["metadata"]["original_height"] == 360
    assert result["processing_status"] == "preprocessed"

    # Verify that the record was saved and can be retrieved.
    record_response = client.get(
        f"/api/v1/records/{result['id']}"
    )

    assert record_response.status_code == 200
    assert record_response.json()["id"] == result["id"]
