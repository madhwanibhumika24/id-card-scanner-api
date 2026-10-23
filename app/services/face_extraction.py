
from pathlib import Path

import cv2
import numpy as np


FACE_DIR = Path("faces")
FACE_DIR.mkdir(parents=True, exist_ok=True)

CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


def extract_face_thumbnail(
    image: np.ndarray,
    image_id: str,
) -> dict:
    """
    Detect a face in an ID-card image and save a thumbnail.

    Returns the saved path and detection status.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image supplied")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = cv2.CascadeClassifier(CASCADE_PATH)

    if detector.empty():
        raise RuntimeError("Could not load OpenCV face detector")

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return {
            "face_detected": False,
            "face_thumbnail": None,
            "message": "No face detected in the uploaded image",
        }

    # Select the largest detected face.
    x, y, width, height = max(
        faces,
        key=lambda face: face[2] * face[3],
    )

    # Add padding around the detected face.
    padding = int(max(width, height) * 0.15)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + width + padding)
    y2 = min(image.shape[0], y + height + padding)

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        return {
            "face_detected": False,
            "face_thumbnail": None,
            "message": "Could not crop the detected face",
        }

    thumbnail = cv2.resize(
        face_crop,
        (160, 160),
        interpolation=cv2.INTER_AREA,
    )

    output_path = FACE_DIR / f"{image_id}_face.jpg"

    if not cv2.imwrite(str(output_path), thumbnail):
        raise RuntimeError("Could not save face thumbnail")

    return {
        "face_detected": True,
        "face_thumbnail": str(output_path),
        "width": 160,
        "height": 160,
        "message": "Face thumbnail saved successfully",
    }
