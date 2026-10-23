# ID Card Scanner API

A RESTful API built using **Python, FastAPI, and OpenCV** to process ID card images, detect card boundaries, extract face thumbnails, and store processing records in SQLite.

## Features

- Upload and validate JPG, JPEG, and PNG images.
- Detect card boundaries and preprocess images using OpenCV.
- Extract face thumbnails using Haar Cascade.
- Store and retrieve processing records using SQLite.
- REST API with interactive Swagger documentation.

**Note:** OCR integration is prepared but currently disabled. Extracted text fields return `null` until OCR is enabled.

## Tech Stack

Python | FastAPI | OpenCV | NumPy | EasyOCR | SQLite | Uvicorn

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/madhwanibhumika24/id-card-scanner-api.git
cd id-card-scanner-api
```

**2. Create and activate a virtual environment**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** to explore and test the API.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/api/v1/process-id` | Upload and process an ID card |
| GET | `/api/v1/records/{record_id}` | Retrieve a saved record |

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

- Enable OCR and structured text extraction.
- Improve card detection accuracy.
- Add authentication and secure file storage.
- Expand automated testing.

## Author

**Bhumika Madhwani**

GitHub: [madhwanibhumika24](https://github.com/madhwanibhumika24)

**Repository:** [id-card-scanner-api](https://github.com/madhwanibhumika24/id-card-scanner-api)
