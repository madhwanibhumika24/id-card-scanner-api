
import json

from app.database.db import get_connection


def save_record(record: dict):
    """Save a successfully processed ID card record."""
    fields = record.get("extracted_fields", {})
    face = record.get("face_extraction", {})

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO id_records (
                id,
                filename,
                full_name,
                document_number,
                date_of_birth,
                expiry_date,
                face_thumbnail,
                metadata,
                preprocessing,
                ocr_text,
                processing_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["filename"],
                fields.get("full_name"),
                fields.get("document_number"),
                fields.get("date_of_birth"),
                fields.get("expiry_date"),
                face.get("face_thumbnail"),
                json.dumps(record.get("metadata", {})),
                json.dumps(record.get("preprocessing", {})),
                record.get("ocr", {}).get("full_text", ""),
                record.get("processing_status", "pending"),
            ),
        )
        connection.commit()


def get_record_by_id(record_id: str):
    """Retrieve a previously saved record by its ID."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT * FROM id_records
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()

    if row is None:
        return None

    result = dict(row)

    result["metadata"] = json.loads(result["metadata"])
    result["preprocessing"] = json.loads(result["preprocessing"])

    result["extracted_fields"] = {
        "full_name": result.pop("full_name"),
        "document_number": result.pop("document_number"),
        "date_of_birth": result.pop("date_of_birth"),
        "expiry_date": result.pop("expiry_date"),
    }

    result["face_extraction"] = {
        "face_thumbnail": result.pop("face_thumbnail"),
    }

    return result
