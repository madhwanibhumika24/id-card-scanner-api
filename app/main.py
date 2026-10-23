
from fastapi import FastAPI

from app.routes.id_routes import router as id_router

app = FastAPI(
    title="ID Card Scanner API",
    description="REST API for ID card image processing and information extraction.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {"message": "ID Card Scanner API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


app.include_router(id_router, prefix="/api/v1", tags=["ID Card Processing"])
