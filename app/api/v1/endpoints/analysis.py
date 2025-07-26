from fastapi import APIRouter, Depends
from app.api.dependencies import get_data_service
from app.services.analysis import analyze_machine_performance, analyze_operator_performance, analyze_temporal_patterns, load_and_prepare_data, plot_correlation_heatmap, plot_failures_per_month, plot_unexpected_stops_heatmaps
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

@router.get("/failures_per_month")
async def analyze_failures():
   data = load_and_prepare_data("data/cleaned_dataset.csv")
   image = plot_failures_per_month(data)
   return JSONResponse({
        "image": f"data:image/png;base64,{image}",
        "message": "Image loaded successfully"
    })

@router.get("/unexpected_stops")
async def unexpected_stops():
   data = load_and_prepare_data("data/cleaned_dataset.csv")
   image = plot_unexpected_stops_heatmaps(data)
   return JSONResponse({
        "image": f"data:image/png;base64,{image}",
        "message": "Image loaded successfully"
    })

@router.get("/analyze_performance")
async def analyze_machine():
   data = load_and_prepare_data("data/cleaned_dataset.csv")
   image = analyze_machine_performance(data)
   return JSONResponse({
        "image": f"data:image/png;base64,{image}",
        "message": "Image loaded successfully"
    })

@router.get("/analyze_operator")
async def analyze_operator():
   data = load_and_prepare_data("data/cleaned_dataset.csv")
   image = analyze_operator_performance(data)
   return JSONResponse({
        "image": f"data:image/png;base64,{image}",
        "message": "Image loaded successfully"
    })

@router.get("/analyze_patterns")
async def analyze_patterns():
   data = load_and_prepare_data("data/cleaned_dataset.csv")
   image = analyze_temporal_patterns(data)
   return JSONResponse({
        "image": f"data:image/png;base64,{image}",
        "message": "Image loaded successfully"
    })

