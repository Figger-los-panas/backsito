from fastapi import APIRouter
from app.api.v1.endpoints import temperature

api_router = APIRouter()

# Include temperature endpoints
api_router.include_router(
    temperature.router,
    prefix="/temperature",
    tags=["temperature"]
)

@api_router.get("/health")
async def health_check():
    """API health check endpoint"""
    return {"status": "healthy", "message": "Temperature API is running"}
