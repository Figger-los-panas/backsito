from app.repositories.csv_repository import CSVRepository
from app.services.data_service import DataService
from fastapi import Depends

def get_csv_repository() -> CSVRepository:
    """Get CSV repository instance"""
    return CSVRepository()

def get_data_service(
    csv_repository: CSVRepository = Depends(get_csv_repository)  # Use Depends()
) -> DataService:
    """Get data service instance"""
    return DataService(csv_repository)
