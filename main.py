import os
import uuid
import yaml
import logging
import filetype 
import shutil
import asyncio
from typing import List, Optional, Dict, Any
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
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.api.schemas import DocumentResponse, ProcessingResult, CorrectionRequest
from src.pipeline import DocumentProcessor
from src.database.db import get_db, init_db, CRUD, SessionLocal, ProcessedDocument
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

# Init Tables
init_db()

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

processor = DocumentProcessor()
try:
    processor.extractor = HybridExtractor()
    logger.info("Swapped default extractor with HybridExtractor.")
except Exception as e:
    logger.error(f"Could not load HybridExtractor: {e}. Using default.")

# Background Task
def process_file_task(doc_id: str, file_path: str):
    # Create a fresh session for the background task
    db = SessionLocal()
    try:
        logger.info(f"Starting processing for {doc_id}")
        result = processor.process_document(file_path)
        
        if result['status'] == 'success':
            CRUD.update_document_result(
                db, 
                doc_id, 
                classification={
                    'document_type': result.get('document_type'),
                    'confidence': result.get('confidence')
                },
                extractions=result.get('extracted_fields', {}),
                text_content=result.get('text_content', "")
            )
            logger.info(f"Processing completed for {doc_id}")
        else:
             CRUD.mark_failed(db, doc_id, result.get('error', 'Unknown error'))
        
    except Exception as e:
        logger.error(f"Processing failed for {doc_id}: {e}")
        CRUD.mark_failed(db, doc_id, str(e))
    finally:
        db.close()

# Endpoints

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/process", response_model=DocumentResponse)
@limiter.limit("10/minute")
async def process_document(
    request: Request,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    # 1. Validation
    contents = await file.read(2048)
    kind = filetype.guess(contents)
    await file.seek(0) # Reset
    
    allowed_mimes = ["application/pdf", "image/jpeg", "image/png", "image/tiff"]
    
    if kind is None:
         if file.content_type not in allowed_mimes:
             raise HTTPException(status_code=400, detail="Unknown file type.")
    elif kind.mime not in allowed_mimes:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {kind.mime}. Allowed: PDF, JPEG, PNG, TIFF")
    
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10 * 1024 * 1024: 
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
    CRUD.create_document(db, doc_id, file.filename)

    # 4. Enqueue
    background_tasks.add_task(process_file_task, doc_id, file_path)

    return DocumentResponse(
        status="success",
        message="Document uploaded and processing started.",
        data={"document_id": doc_id, "status": "processing"}
    )

@app.get("/api/v1/result/{document_id}", response_model=DocumentResponse)
def get_result(document_id: str, db: Session = Depends(get_db)):
    doc = CRUD.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    res = ProcessingResult(
        document_id=doc['id'],
        filename=doc['filename'],
        status=doc['status'],
        upload_timestamp=datetime.fromisoformat(doc['upload_timestamp']) if doc['upload_timestamp'] else datetime.utcnow(),
        processed_timestamp=datetime.fromisoformat(doc['processed_timestamp']) if doc.get('processed_timestamp') else None,
        extracted_data=doc['result_json'].get('extracted_fields'),
        document_type=doc['result_json'].get('document_type'),
        confidence=doc['result_json'].get('confidence'),
        error=doc['result_json'].get('error')
    )
    
    return DocumentResponse(
        status="success",
        message="Result retrieved",
        data=res
    )

@app.get("/api/v1/documents")
def get_recent_documents(limit: int = 10, db: Session = Depends(get_db)):
    docs = db.query(ProcessedDocument).order_by(desc(ProcessedDocument.upload_timestamp)).limit(limit).all()
    
    results = []
    for doc in docs:
        results.append({
            "id": doc.id,
            "filename": doc.filename,
            "status": doc.status,
            "upload_timestamp": doc.upload_timestamp.isoformat() if doc.upload_timestamp else None
        })
        
    return DocumentResponse(
        status="success",
        message="Recent documents retrieved",
        data=results
    )

@app.get("/api/v1/export/{document_id}")
def export_data(document_id: str, format: str = "json", db: Session = Depends(get_db)):
    doc = CRUD.get_document(db, document_id)
    if not doc or doc['status'] != 'completed':
        raise HTTPException(status_code=404, detail="Document not found or not ready")
    
    data = doc['result_json'].get('extracted_fields', {})
    
    if format.lower() == 'json':
        return JSONResponse(content=data)
    elif format.lower() == 'csv':
        flat_data = {}
        for k, v in data.items():
            if isinstance(v, list):
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
def submit_correction(document_id: str, request: CorrectionRequest, db: Session = Depends(get_db)):
    doc = CRUD.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    CRUD.add_correction(db, document_id, request.updates)
    
    return DocumentResponse(
        status="success",
        message="Corrections applied successfully",
        data={"document_id": document_id}
    )
