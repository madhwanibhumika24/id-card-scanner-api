
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.image_processing import preprocess_image

router = APIRouter()

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post("/process-id")
async def process_id(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, and PNG images are allowed",
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Maximum file size is 10 MB")

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a readable image",
        )

    height, width = image.shape[:2]

    if width < 100 or height < 100:
        raise HTTPException(
            status_code=400,
            detail="Image must be at least 100 x 100 pixels",
        )

    image_id = str(uuid4())
    upload_path = UPLOAD_DIR / f"{image_id}{extension}"
    processed_path = PROCESSED_DIR / f"{image_id}.jpg"

    if not cv2.imwrite(str(upload_path), image):
        raise HTTPException(status_code=500, detail="Could not save upload")

    try:
        result = preprocess_image(image, processed_path)
    except (ValueError, RuntimeError, cv2.error) as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"Image preprocessing failed: {exc}",
        ) from exc

    return {
        "id": image_id,
        "message": "Image uploaded and preprocessed successfully",
        "metadata": {
            "original_width": width,
            "original_height": height,
            "format": extension.lstrip("."),
            "size_bytes": len(contents),
        },
        "preprocessing": result,
        "extracted_fields": {
            "full_name": None,
            "document_number": None,
            "date_of_birth": None,
            "expiry_date": None,
        },
        "processing_status": "preprocessed",
    }
