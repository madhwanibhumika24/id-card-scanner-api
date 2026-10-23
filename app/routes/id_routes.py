
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


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
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MB",
        )

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
            detail="Image dimensions must be at least 100 x 100 pixels",
        )

    image_id = str(uuid4())
    saved_path = UPLOAD_DIR / f"{image_id}{extension}"

    if not cv2.imwrite(str(saved_path), image):
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded image",
        )

    return {
        "id": image_id,
        "message": "Image uploaded successfully",
        "filename": saved_path.name,
        "metadata": {
            "width": width,
            "height": height,
            "format": extension.lstrip("."),
            "size_bytes": len(contents),
        },
        "extracted_fields": {
            "full_name": None,
            "document_number": None,
            "date_of_birth": None,
            "expiry_date": None,
        },
        "processing_status": "uploaded"
    }
