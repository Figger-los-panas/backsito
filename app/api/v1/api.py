from app.api.v1.endpoints import temperature, analysis
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import json
from pathlib import Path

api_router = APIRouter()

# Include temperature endpoints
api_router.include_router(
    temperature.router,
    prefix="/temperature",
    tags=["temperature"]
)

api_router.include_router(
   analysis.router,
   prefix="/analysis",
   tags=["analysis"]
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


@api_router.get("/spearman")
async def spearman():
    """API SPEARMAN IMAGE"""
    try:
        file_path = Path(f"data/spearman.txt")
        
        # Read the base64 string from the txt file
        with open(file_path, "r") as f:
            base64_string = f.read().strip()
        
        # Return with proper data URL prefix
        return JSONResponse({
            "image": f"data:image/png;base64,{base64_string}",
            "message": "Image loaded successfully"
        })
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Data file not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON format")
