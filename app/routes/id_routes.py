
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.database.models import save_record, get_record_by_id
from app.services.image_processing import preprocess_image
from app.services.face_extraction import extract_face_thumbnail

router = APIRouter()

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
FACE_DIR = Path("faces")

for directory in (UPLOAD_DIR, PROCESSED_DIR, FACE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post("/process-id")
async def process_id(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, and PNG images are allowed",
        )

    contents = await file.read(MAX_FILE_SIZE + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 10 MB",
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
            detail="Image must be at least 100 x 100 pixels",
        )

    image_id = str(uuid4())
    safe_filename = f"{image_id}{extension}"

    upload_path = UPLOAD_DIR / safe_filename
    processed_path = PROCESSED_DIR / f"{image_id}.jpg"
    ocr_image_path = PROCESSED_DIR / f"{image_id}_ocr.png"
    face_path = FACE_DIR / f"{image_id}_face.jpg"

    if not cv2.imwrite(str(upload_path), image):
        raise HTTPException(
            status_code=500,
            detail="Could not save uploaded image",
        )

    try:
        preprocessing = preprocess_image(image, processed_path)
        face_result = extract_face_thumbnail(image, image_id)

        # OCR is temporarily disabled until the model download is fixed.
        ocr_result = {
            "full_text": "",
            "lines": [],
        }

        extracted_fields = {
            "full_name": None,
            "document_number": None,
            "date_of_birth": None,
            "expiry_date": None,
        }

        record = {
            "id": image_id,
            "filename": safe_filename,
            "metadata": {
                "original_width": width,
                "original_height": height,
                "format": extension.lstrip("."),
                "size_bytes": len(contents),
            },
            "preprocessing": preprocessing,
            "face_extraction": face_result,
            "extracted_fields": extracted_fields,
            "ocr": ocr_result,
            "processing_status": "preprocessed",
        }

        save_record(record)

    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        processed_path.unlink(missing_ok=True)
        ocr_image_path.unlink(missing_ok=True)
        face_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Image processing or database save failed: {exc}",
        ) from exc

    return {
        **record,
        "message": "Image processed and record saved successfully",
    }


@router.get("/records/{record_id}")
def get_record(record_id: str):
    """Retrieve a previously processed record from SQLite."""
    record = get_record_by_id(record_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Record not found",
        )

    return record
