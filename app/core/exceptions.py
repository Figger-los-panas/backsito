import logging
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

logger = logging.getLogger(__name__)

class DataProcessingError(Exception):
    """Raised when data processing fails"""
    pass

class AIServiceError(Exception):
    """Raised when AI service fails"""
    pass

class FileNotFoundError(Exception):
    """Raised when CSV file is not found"""
    pass


def setup_exception_handlers(app):
    @app.exception_handler(DataProcessingError)
    async def data_processing_exception_handler(request: Request, exc: DataProcessingError):
        logger.error(f"Data processing error: {exc}")
        return JSONResponse(
            status_code=422,
            content={"message": "Error processing data", "detail": str(exc)}
        )
    
    @app.exception_handler(AIServiceError)
    async def ai_service_exception_handler(request: Request, exc: AIServiceError):
        logger.error(f"AI service error: {exc}")
        return JSONResponse(
            status_code=503,
            content={"message": "AI service temporarily unavailable", "detail": str(exc)}
        )
    
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_exception_handler(request: Request, exc: FileNotFoundError):
        logger.error(f"File not found: {exc}")
        return JSONResponse(
            status_code=404,
            content={"message": "Data file not found", "detail": str(exc)}
        )
