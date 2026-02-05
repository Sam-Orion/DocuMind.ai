import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.db import Base, ProcessedDocument, Extraction, Correction, CRUD

# In-memory DB for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

def test_create_document(db_session):
    doc = CRUD.create_document(db_session, "test-id", "test.pdf")
    assert doc.id == "test-id"
    assert doc.filename == "test.pdf"
    assert doc.status == "processing"

def test_update_document_result(db_session):
    CRUD.create_document(db_session, "test-id", "test.pdf")
    
    classification = {"document_type": "invoice", "confidence": 0.95}
    extractions = {
        "total": {"value": "100.00", "confidence": 0.9},
        "date": [{"value": "2023-01-01", "confidence": 1.0}]
    }
    
    doc = CRUD.update_document_result(db_session, "test-id", classification, extractions, "OCR TEXT")
    
    assert doc.status == "completed"
    assert doc.document_type == "invoice"
    assert len(doc.extractions) == 2
    
    # Check extraction values
    val_map = {e.field_key: e.value for e in doc.extractions}
    assert val_map["total"] == "100.00"
    assert val_map["date"] == "2023-01-01"

def test_add_correction(db_session):
    CRUD.create_document(db_session, "test-id", "test.pdf")
    # Initial data
    CRUD.update_document_result(
        db_session, "test-id", {}, {"total": "100.00"}, ""
    )
    
    # Correction
    CRUD.add_correction(db_session, "test-id", {"total": "500.00"})
    
    doc = db_session.query(ProcessedDocument).filter_by(id="test-id").first()
    # Should have 1 extraction now (updated)
    assert len(doc.extractions) == 1
    assert doc.extractions[0].value == "500.00"
    assert doc.extractions[0].source == "manual_correction"
    
    # Should have 1 correction record
    assert len(doc.corrections) == 1
    assert doc.corrections[0].previous_value == "100.00"
    assert doc.corrections[0].new_value == "500.00"
