
from pathlib import Path

import easyocr
from PIL import Image

_reader = None


def get_reader():
    """Load the OCR model once and reuse it."""
    global _reader

    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)

    return _reader


def extract_text(image_path: str) -> dict:
    """Extract text and JSON-compatible OCR results."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("The supplied file is not a valid image.") from exc

    reader = get_reader()
    results = reader.readtext(
        str(path),
        detail=1,
        paragraph=False,
    )

    lines = []

    for bounding_box, text, confidence in results:
        # Convert NumPy coordinates to regular Python floats.
        json_box = [
            [float(point[0]), float(point[1])]
            for point in bounding_box
        ]

        lines.append({
            "text": str(text),
            "confidence": float(round(float(confidence), 4)),
            "bounding_box": json_box,
        })

    full_text = "\n".join(line["text"] for line in lines)

    return {
        "full_text": full_text,
        "lines": lines,
    }
