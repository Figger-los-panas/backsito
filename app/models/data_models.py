from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class ManufacturingDataPoint(BaseModel):
    """Single manufacturing data point from CSV"""
    timestamp: str
    turno: str
    operador_id: str
    maquina_id: str
    producto_id: str
    temperatura: Optional[float]
    vibracion: Optional[float]
    humedad: Optional[float]
    tiempo_ciclo: Optional[float]
    fallo_detectado: str
    tipo_fallo: Optional[str]
    cantidad_producida: Optional[int]
    unidades_defectuosas: Optional[int]
    eficiencia_porcentual: Optional[float]
    consumo_energia: Optional[float]
    paradas_programadas: Optional[int]
    paradas_imprevistas: Optional[int]
    observaciones: Optional[str]

class DataPoint(BaseModel):
    index: int
    values: Dict[str, Any]

class TemperaturePoint(BaseModel):
    timestamp: str
    machine_id: str
    temperature: float

class TemperatureResponse(BaseModel):
    data: List[TemperaturePoint]
    total_records: int

class TemperatureFilterRequest(BaseModel):
    """Request model for filtering temperature data"""
    machine_ids: Optional[List[str]] = None
    start_date: Optional[str] = None  # "2023-01-01"
    end_date: Optional[str] = None    # "2023-01-02"
    limit: Optional[int] = 1000
