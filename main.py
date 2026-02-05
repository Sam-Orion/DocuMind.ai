import os
import uuid
import yaml
import logging
import filetype 
import shutil
import asyncio
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import pandas as pd
import io
import json

from src.api.schemas import DocumentResponse, ProcessingResult, CorrectionRequest
from src.pipeline import DocumentProcessor
from src.database.db import Database
from src.extraction.hybrid_extractor import HybridExtractor 

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
UPLOAD_DIR = "data/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Components
app = FastAPI(
    title="DocuMind AI API",
    description="Intelligent Document Processing API",
    version="0.1.0"
)

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
db = Database()

processor = DocumentProcessor()
try:
    processor.extractor = HybridExtractor()
    logger.info("Swapped default extractor with HybridExtractor.")
except Exception as e:
    logger.error(f"Could not load HybridExtractor: {e}. Using default.")

# Background Task
def process_file_task(doc_id: str, file_path: str):
    try:
        logger.info(f"Starting processing for {doc_id}")
        result = processor.process_document(file_path)
        db.update_result(doc_id, result, status="completed")
        logger.info(f"Processing completed for {doc_id}")
        
    except Exception as e:
        logger.error(f"Processing failed for {doc_id}: {e}")
        db.update_result(doc_id, {"error": str(e)}, status="failed")

# Endpoints

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/process", response_model=DocumentResponse)
@limiter.limit("10/minute")
async def process_document(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    # 1. Validation
    # Read file header for kind check (first 2kb)
    contents = await file.read(2048)
    kind = filetype.guess(contents)
    await file.seek(0) # Reset
    
    # MIME Checks
    # Allowed: pdf, jpg, png, tiff
    # filetype returns validation object
    
    allowed_mimes = ["application/pdf", "image/jpeg", "image/png", "image/tiff"]
    
    if kind is None:
         # Fallback check methods or strict fail? 
         # Some PDFs might not match magic bytes easily if malformed?
         # Check extension as fallback context
         if file.content_type not in allowed_mimes:
             raise HTTPException(status_code=400, detail="Unknown file type.")
    elif kind.mime not in allowed_mimes:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {kind.mime}. Allowed: PDF, JPEG, PNG, TIFF")
    
    # Check size
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10 * 1024 * 1024: # 10MB
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    # 2. Save
    doc_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    filename = f"{doc_id}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    # 3. DB Init
    db.save_document(doc_id, file.filename)

    # 4. Enqueue
    background_tasks.add_task(process_file_task, doc_id, file_path)

    return DocumentResponse(
        status="success",
        message="Document uploaded and processing started.",
        data={"document_id": doc_id, "status": "processing"}
    )

@app.get("/api/v1/result/{document_id}", response_model=DocumentResponse)
def get_result(document_id: str):
    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Construct response
    res = ProcessingResult(
        document_id=doc['id'],
        filename=doc['filename'],
        status=doc['status'],
        upload_timestamp=datetime.fromisoformat(doc['upload_timestamp']),
        processed_timestamp=datetime.fromisoformat(doc['processed_timestamp']) if doc['processed_timestamp'] else None,
        extracted_data=doc['result_json'].get('extracted_fields') if doc['result_json'] else None,
        document_type=doc['result_json'].get('document_type'),
        confidence=doc['result_json'].get('confidence'),
        error=doc['result_json'].get('error')
    )
    
    return DocumentResponse(
        status="success",
        message="Result retrieved",
        data=res
    )

@app.get("/api/v1/export/{document_id}")
def export_data(document_id: str, format: str = "json"):
    doc = db.get_document(document_id)
    if not doc or doc['status'] != 'completed':
        raise HTTPException(status_code=404, detail="Document not found or not ready")
    
    data = doc['result_json'].get('extracted_fields', {})
    
    # Flatten specific complex fields if CSV
    
    if format.lower() == 'json':
        return JSONResponse(content=data)
    elif format.lower() == 'csv':
        # Simple flattening for MVP: keys -> cols
        # Complex lists (like line items) might need special handling.
        # For now, just top-level kv pairs.
        flat_data = {}
        for k, v in data.items():
            if isinstance(v, list):
                 # Join first few or just count? 
                 # e.g. emails: "a@b.com; c@d.com"
                 if len(v) > 0 and isinstance(v[0], dict) and 'value' in v[0]:
                     flat_data[k] = "; ".join([str(item['value']) for item in v])
                 else:
                     flat_data[k] = str(v)
            else:
                flat_data[k] = str(v)
                
        df = pd.DataFrame([flat_data])
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = f"attachment; filename=export_{document_id}.csv"
        return response
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use 'json' or 'csv'.")

@app.post("/api/v1/correct/{document_id}", response_model=DocumentResponse)
def submit_correction(document_id: str, request: CorrectionRequest):
    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    current_result = doc['result_json']
    # Merge updates
    # Logic: Update extracted_fields directly?
    
    if 'extracted_fields' not in current_result:
        current_result['extracted_fields'] = {}
        
    # We update the fields. Pydantic ensures 'updates' is a dict.
    # Note: A real system would track version history.
    for k, v in request.updates.items():
        # User provides corrected values (simple strings usually)
        # We might wrap them back into our dict format with "confidence": 1.0 (Manual)
        
        # Check if user passed full structure or just value
        if isinstance(v, dict) and 'value' in v:
            current_result['extracted_fields'][k] = [v] # Replace or append? Assuming replace for simple fields
        else:
             # Assume single value replacement for simple fields
             # We need to wrap it to match our 'List[Dict]' structure if standardizing
             # Or just allow mixed types.
             # For consistency with pipeline:
             current_result['extracted_fields'][k] = [{
                 "value": v,
                 "confidence": 1.0,
                 "source": "manual_correction"
             }]
             
    db.update_result(document_id, current_result, status="completed")
    
    return DocumentResponse(
        status="success",
        message="Corrections applied successfully",
        data={"document_id": document_id}
    )
