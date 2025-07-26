from fastapi import APIRouter, Depends
from app.models.data_models import TemperatureResponse, TemperatureFilterRequest
from app.services.data_service import DataService
from app.api.dependencies import get_data_service
from typing import List

router = APIRouter()
@router.post("/", response_model=TemperatureResponse, summary="Get  temperature per machine over time")
async def get_temperature_data(
        filter_request: TemperatureFilterRequest,
        data_service: DataService = Depends(get_data_service)
        ):
    return data_service.get_temperature_data(filter_request)

@router.get("/machines", response_model=List[str], summary="Get available machine IDs")
async def get_machines(
    data_service: DataService = Depends(get_data_service)
):
    """Get list of all available machine IDs in the dataset"""
    return data_service.get_machines()
