from app.api.v1.endpoints import temperature
from fastapi import APIRouter, HTTPException
import json
from pathlib import Path

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

@api_router.get("/combinations")
async def combinations():
    """API COMBINATIONS JSON"""
    try:
        file_path = Path("data/optimal_combinations.json")
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Data file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON format")
