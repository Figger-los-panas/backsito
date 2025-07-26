from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional, List

class Settings(BaseSettings):
    app_name: str = "Cigger API"
    debug: bool = True
    version: str = "1.0.0"
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"  # Keep as string
    csv_file_path: str = "data/sample_data.csv"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # Convert to list when you need it
    def get_allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(',')]
    
    class Config:
        env_file = ".env"

settings = Settings()
