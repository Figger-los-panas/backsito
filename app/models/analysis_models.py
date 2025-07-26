from pydantic import BaseModel
from typing import Dict, Any

class AnalsisResponse(BaseModel):
    data_summary: Dict[str, Any]
