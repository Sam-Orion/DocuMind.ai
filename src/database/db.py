from sqlalchemy import create_engine, Column, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from datetime import datetime, timezone
import json
import uuid
from typing import Dict, Any, List, Optional

# Constants
DATABASE_URL = "sqlite:///./documind.db"

# Setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class ProcessedDocument(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    status = Column(String, default="processing")
    upload_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    processed_timestamp = Column(DateTime, nullable=True)
    document_type = Column(String, nullable=True)
    text_content = Column(Text, nullable=True) # Full OCR text
    confidence = Column(Float, nullable=True)
    error = Column(String, nullable=True)
    
    # Relationships
    extractions = relationship("Extraction", back_populates="document", cascade="all, delete-orphan")
    corrections = relationship("Correction", back_populates="document", cascade="all, delete-orphan")

class Extraction(Base):
    __tablename__ = "extractions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"))
    field_key = Column(String) # e.g. "email", "total_amount"
    value = Column(String) # Stored as string, convert if needed
    confidence = Column(Float)
    source = Column(String) # "regex", "spacy", "hybrid"
    position_json = Column(String, nullable=True) # JSON dump of position
    
    document = relationship("ProcessedDocument", back_populates="extractions")

class Correction(Base):
    __tablename__ = "corrections"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"))
    field_key = Column(String)
    previous_value = Column(String, nullable=True)
    new_value = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    document = relationship("ProcessedDocument", back_populates="corrections")

# Init DB
def init_db():
    Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# CRUD Utilities
class CRUD:
    @staticmethod
    def create_document(db: Session, doc_id: str, filename: str) -> ProcessedDocument:
        db_doc = ProcessedDocument(id=doc_id, filename=filename, status="processing")
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        return db_doc

    @staticmethod
    def update_document_result(db: Session, doc_id: str, classification: Dict[str, Any], extractions: Dict[str, Any], text_content: str):
        db_doc = db.query(ProcessedDocument).filter(ProcessedDocument.id == doc_id).first()
        if not db_doc:
            return
            
        db_doc.status = "completed"
        db_doc.processed_timestamp = datetime.now(timezone.utc)
        db_doc.document_type = classification.get('document_type')
        db_doc.confidence = classification.get('confidence')
        db_doc.text_content = text_content
        
        # Save Extractions
        # Extractions input format: {"email": [{"value": "x", "confidence": 0.9, ...}], "total": ...}
        for key, items in extractions.items():
            if not isinstance(items, list):
                items = [items] # Normalize
                
            for item in items:
                if not isinstance(item, dict): 
                    # Simple val
                    val = str(item)
                    conf = 1.0
                    src = "unknown"
                    pos = None
                else:
                    val = str(item.get('value', ''))
                    conf = float(item.get('confidence', 1.0))
                    src = item.get('source', 'hybrid')
                    pos = json.dumps(item.get('position')) if item.get('position') else None
                
                db_extraction = Extraction(
                    document_id=doc_id,
                    field_key=key,
                    value=val,
                    confidence=conf,
                    source=src,
                    position_json=pos
                )
                db.add(db_extraction)
        
        db.commit()
        db.refresh(db_doc)
        return db_doc
        
    @staticmethod
    def mark_failed(db: Session, doc_id: str, error: str):
        db_doc = db.query(ProcessedDocument).filter(ProcessedDocument.id == doc_id).first()
        if db_doc:
            db_doc.status = "failed"
            db_doc.error = error
            db.commit()

    @staticmethod
    def get_document(db: Session, doc_id: str) -> Optional[Dict[str, Any]]:
        db_doc = db.query(ProcessedDocument).filter(ProcessedDocument.id == doc_id).first()
        if not db_doc:
            return None
            
        # Reconstruct Dictionary structure for API response compatibility
        extractions_dict = {}
        for ext in db_doc.extractions:
            if ext.field_key not in extractions_dict:
                extractions_dict[ext.field_key] = []
            
            # Simple dict representation
            extractions_dict[ext.field_key].append({
                "value": ext.value,
                "confidence": ext.confidence
            })
            
        return {
            "id": db_doc.id,
            "filename": db_doc.filename,
            "status": db_doc.status,
            "upload_timestamp": db_doc.upload_timestamp.isoformat() if db_doc.upload_timestamp else None,
            "processed_timestamp": db_doc.processed_timestamp.isoformat() if db_doc.processed_timestamp else None,
            "result_json": { # Reconstruct legacy structure
                "document_type": db_doc.document_type,
                "confidence": db_doc.confidence,
                "extracted_fields": extractions_dict,
                "error": db_doc.error
            }
        }
    
    @staticmethod
    def add_correction(db: Session, doc_id: str, updates: Dict[str, Any]):
        # For simplicity, we create a Correction record AND update the Extraction
        # If extraction doesn't exist for that key, create it
        
        for key, value in updates.items():
            # Find existing extraction logic: assuming single value per key for MVP correction
            # If multiple exist, we delete/overwrite or pick first?
            # Strategy: Delete all existing extractions for this key, insert new one (User Override)
            
            existing = db.query(Extraction).filter(Extraction.document_id == doc_id, Extraction.field_key == key).all()
            
            # Record Correction History (just logging the user action roughly)
            prev_val = existing[0].value if existing else None
            
            # Normalize value
            if isinstance(value, dict) and 'value' in value:
                new_val_str = str(value['value'])
            else:
                new_val_str = str(value)

            corr = Correction(
                document_id=doc_id,
                field_key=key,
                previous_value=prev_val,
                new_value=new_val_str
            )
            db.add(corr)
            
            # Update Extraction State
            # Delete old
            for ext in existing:
                db.delete(ext)
            
            # Add new (Manual)
            new_ext = Extraction(
                document_id=doc_id,
                field_key=key,
                value=new_val_str,
                confidence=1.0,
                source="manual_correction"
            )
            db.add(new_ext)
            
        db.commit()
