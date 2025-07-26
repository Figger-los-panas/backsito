from typing import List
from app.repositories.csv_repository import CSVRepository
from app.models.data_models import TemperatureResponse, TemperaturePoint, TemperatureFilterRequest
from app.core.exceptions import DataProcessingError

class DataService:
    def __init__(self, csv_repository: CSVRepository):
        self.csv_repository = csv_repository
    
    def get_temperature_data(self, filter_request: TemperatureFilterRequest) -> TemperatureResponse:
        """Get temperature data per machine over time"""
        try:
            df = self.csv_repository.get_temperature_data(
                machine_ids=filter_request.machine_ids,
                start_date=filter_request.start_date,
                end_date=filter_request.end_date,
                limit=filter_request.limit
            )
            
            # Convert to response format
            temperature_points = []
            for _, row in df.iterrows():
                point = TemperaturePoint(
                    timestamp=row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    machine_id=row['maquina_id'],
                    temperature=float(row['temperatura'])
                )
                temperature_points.append(point)
            
            return TemperatureResponse(
                data=temperature_points,
                total_records=len(temperature_points)
            )
        
        except Exception as e:
            raise DataProcessingError(f"Error retrieving temperature data: {str(e)}")
    
    def get_machines(self) -> List[str]:
        """Get list of available machine IDs"""
        return self.csv_repository.get_unique_machines()

    def get_all_data(self):
        return self.csv_repository.get_all_data()
