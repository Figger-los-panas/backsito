import pandas as pd
from app.config import settings
from app.core.exceptions import DataProcessingError, FileNotFoundError
from typing import List, Optional
import os

class CSVRepository:
    def __init__(self):
        self.file_path = settings.csv_file_path
        self._df: Optional[pd.DataFrame] = None

    def _load_data(self) -> pd.DataFrame:
        """Load CSV data with caching"""
        if self._df is None:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"CSV file not found: {self.file_path}")
            
            try:
                self._df = pd.read_csv(self.file_path)
                # Convert timestamp to datetime
                self._df['timestamp'] = pd.to_datetime(self._df['timestamp'])
            except Exception as e:
                raise DataProcessingError(f"Error reading CSV file: {str(e)}")
        
        return self._df

    def get_all_data(self) -> pd.DataFrame:
        """Get all data from CSV"""
        return self._load_data()
    
    def get_columns(self) -> List[str]:
        """Get column names"""
        df = self._load_data()
        return df.columns.tolist()

    def get_temperature_data(
        self,
        machine_ids: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get temperature data with filters"""
        df = self._load_data()
        
        # Select only needed columns and filter out null temperatures
        temp_df = df[['timestamp', 'maquina_id', 'temperatura']].copy()
        temp_df = temp_df.dropna(subset=['temperatura'])
        
        # Apply filters
        if machine_ids:
            temp_df = temp_df[temp_df['maquina_id'].isin(machine_ids)]
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            temp_df = temp_df[temp_df['timestamp'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            temp_df = temp_df[temp_df['timestamp'] <= end_dt]
        
        # Sort by timestamp
        temp_df = temp_df.sort_values('timestamp')
        
        # Apply limit
        if limit:
            temp_df = temp_df.head(limit)
        
        return temp_df

    def get_unique_machines(self) -> List[str]:
        """Get list of unique machine IDs"""
        df = self._load_data()
        return df['maquina_id'].unique().tolist()
