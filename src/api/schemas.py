from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

class ProcessingResult(BaseModel):
    document_id: str
    filename: str
    status: str
    upload_timestamp: datetime
    processed_timestamp: Optional[datetime] = None
    document_type: Optional[str] = None
    confidence: Optional[float] = None
    extracted_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class DocumentResponse(BaseModel):
    status: str
    message: str
    data: Optional[Union[ProcessingResult, Dict, str]] = None

class CorrectionRequest(BaseModel):
    updates: Dict[str, Any]
