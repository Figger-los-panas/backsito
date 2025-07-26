from fastapi import APIRouter, Depends
from app.api.dependencies import get_data_service
from app.services.analysis import plot_correlation_heatmap
from app.services.data_service import DataService
from fastapi.responses import JSONResponse


router = APIRouter()
@router.get("/spearman")
async def analyze_spearman(data_service: DataService = Depends(get_data_service)):
   data = data_service.get_all_data()     
   image = plot_correlation_heatmap(data)
   return JSONResponse({
        "image": f"data:image/png;base64,{image}",
        "message": "Image loaded successfully"
    })


