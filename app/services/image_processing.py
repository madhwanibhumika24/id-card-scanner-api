
from pathlib import Path

import cv2
import numpy as np


PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def order_points(points: np.ndarray) -> np.ndarray:
    """Order corners as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)

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
    """Correct perspective using four card corners."""
    top_left, top_right, bottom_right, bottom_left = order_points(points)

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)

    max_width = max(1, int(max(width_a, width_b)))
    max_height = max(1, int(max(height_a, height_b)))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        order_points(points),
        destination,
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height),
    )


def detect_card_boundary(image: np.ndarray) -> np.ndarray | None:
    """Find a plausible, large four-corner card boundary."""
    height, width = image.shape[:2]
    image_area = height * width

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edge_maps = [
        cv2.Canny(blurred, 30, 100),
        cv2.Canny(blurred, 50, 150),
    ]

    candidates = []

    for edges in edge_maps:
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            # Ignore tiny text and face contours.
            if area < image_area * 0.10:
                continue

            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True,
            )

            if len(approximation) != 4:
                continue

            points = approximation.reshape(4, 2).astype(np.float32)

            if not cv2.isContourConvex(approximation):
                continue

            ordered = order_points(points)

            top_left, top_right, bottom_right, bottom_left = ordered

            card_width = max(
                np.linalg.norm(top_right - top_left),
                np.linalg.norm(bottom_right - bottom_left),
            )
            card_height = max(
                np.linalg.norm(bottom_left - top_left),
                np.linalg.norm(bottom_right - top_right),
            )

            if min(card_width, card_height) < 50:
                continue

            ratio = max(card_width, card_height) / max(
                min(card_width, card_height), 1
            )

            # Typical ID cards are landscape, though portrait cards exist.
            if not 1.2 <= ratio <= 2.2:
                continue

            candidates.append((area, points))

    if not candidates:
        return None

    # Prefer the largest plausible card contour.
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def preprocess_image(
    image: np.ndarray,
    output_path: str | Path,
) -> dict:
    """Detect, align, and prepare an ID card for OCR."""
    if image is None or image.size == 0:
        raise ValueError("Invalid image supplied")

    corners = detect_card_boundary(image)

    if corners is not None:
        card = four_point_transform(image, corners)
        detection_method = "contour"
    else:
        # Avoid cropping to a random small contour when detection fails.
        card = image.copy()
        detection_method = "full_image_fallback"

    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

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
