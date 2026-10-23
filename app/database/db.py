
import sqlite3
from pathlib import Path

DATABASE_PATH = Path("id_scanner.db")


def get_connection():
    """Create and return a SQLite database connection."""
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    """Create the records table if it does not exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS id_records (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                full_name TEXT,
                document_number TEXT,
                date_of_birth TEXT,
                expiry_date TEXT,
                face_thumbnail TEXT,
                metadata TEXT NOT NULL,
                preprocessing TEXT NOT NULL,
                ocr_text TEXT,
                processing_status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()
