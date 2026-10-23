
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database.db import init_db
from app.routes.id_routes import router as id_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="ID Card Scanner API",
    description="REST API for ID card processing and information extraction.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {"message": "ID Card Scanner API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


app.include_router(
    id_router,
    prefix="/api/v1",
    tags=["ID Card Processing"],
)
