
from pathlib import Path

import cv2
import numpy as np


PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four corner points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)

    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(differences)]
    rect[3] = points[np.argmax(differences)]

    return rect


def four_point_transform(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Correct perspective using the four corners of a detected card."""
    rect = order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    max_width = max(int(max(width_a, width_b)), 1)

    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_height = max(int(max(height_a, height_b)), 1)

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)

    return cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
    )


def detect_card_boundary(image: np.ndarray) -> np.ndarray | None:
    """Detect a likely four-corner document boundary."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    image_area = image.shape[0] * image.shape[1]

    for contour in contours[:20]:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        if len(approximation) == 4:
            area = cv2.contourArea(approximation)

            if image_area * 0.05 <= area <= image_area * 0.98:
                return approximation.reshape(4, 2).astype(np.float32)

    return None


def preprocess_image(
    image: np.ndarray,
    output_path: str | Path,
) -> dict:
    """Detect, align, and prepare an ID card image for OCR."""
    if image is None or image.size == 0:
        raise ValueError("Invalid image supplied")

    corners = detect_card_boundary(image)

    if corners is not None:
        card = four_point_transform(image, corners)
        detection_method = "contour"
    else:
        # Fallback: use the complete image if no reliable boundary is found.
        card = image.copy()
        detection_method = "full_image_fallback"

    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive thresholding creates a high-contrast OCR image.
    threshold = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), card):
        raise RuntimeError("Could not save aligned card image")

    threshold_path = output_path.with_name(
        f"{output_path.stem}_ocr.png"
    )

    if not cv2.imwrite(str(threshold_path), threshold):
        raise RuntimeError("Could not save OCR preprocessing image")

    return {
        "processed_path": str(output_path),
        "ocr_image_path": str(threshold_path),
        "detection_method": detection_method,
        "width": int(card.shape[1]),
        "height": int(card.shape[0]),
    }
