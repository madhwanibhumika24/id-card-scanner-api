# ID Card Scanner API

A RESTful API built with **Python, FastAPI, OpenCV, and EasyOCR** to process ID card images, detect card boundaries, extract face thumbnails, recognize text, and store processing records in SQLite.

## Features

- Upload and validate JPG, JPEG, and PNG images.
- Validate file size and image dimensions.
- Detect card boundaries and preprocess images using OpenCV.
- Extract face thumbnails using OpenCV Haar Cascade.
- Recognize text using EasyOCR.
- Extract candidate fields such as name, document number, date of birth, and expiry date.
- Store and retrieve processing records using SQLite.
- Provide interactive API documentation through Swagger UI.
- Run automated API tests using pytest.

**Note:** OCR and field extraction accuracy depend on image quality, card layout, and text recognition. Fields that cannot be confidently identified may return `null`.

## Tech Stack

Python | FastAPI | OpenCV | NumPy | EasyOCR | SQLite | Uvicorn | pytest

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/madhwanibhumika24/id-card-scanner-api.git
cd id-card-scanner-api
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the application

```powershell
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to explore and test the API.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/api/v1/process-id` | Upload and process an ID card |
| GET | `/api/v1/records/{record_id}` | Retrieve a saved record |

## Run Tests

```powershell
python -m pytest tests/test_api.py -v
```

The current automated test suite contains six tests covering API status, health checks, invalid uploads, missing records, and image processing and retrieval.

## Project Structure

```text
app/
├── main.py
├── routes/
├── services/
├── database/
└── utils/
uploads/
processed/
faces/
tests/
requirements.txt
README.md
```

## Future Improvements

- Improve card-boundary detection for different image orientations.
- Improve field extraction across different ID-card layouts.
- Add authentication and access control.
- Strengthen file storage security and error logging.
- Expand automated tests for edge cases and OCR accuracy.

## Author

**Bhumika Madhwani**

GitHub: https://github.com/madhwanibhumika24

Repository: https://github.com/madhwanibhumika24/id-card-scanner-api