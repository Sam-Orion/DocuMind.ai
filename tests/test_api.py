import pytest
from fastapi.testclient import TestClient
from main import app, get_db
from src.database.db import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import shutil
import time

from sqlalchemy.pool import StaticPool

# Use in-memory DB for integration tests too
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# Helper to clean up DB/files after tests
@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    # Setup
    Base.metadata.create_all(bind=engine)
    os.makedirs("data/raw", exist_ok=True)
    yield
    # Teardown
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("documind.db"):
        os.remove("documind.db")

    if os.path.exists("data/raw"):
        # Don't delete entire dir if used by others, but for test isolation it's good
        pass 

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_process_document_invalid_file():
    # Test uploading a text file (not allowed)
    files = {'file': ('test.txt', b'This is a text file', 'text/plain')}
    response = client.post("/api/v1/process", files=files)
    assert response.status_code == 400
    # "Unknown file type" is returned when filetype.guess fails to find a magic number match
    assert "Unknown file type" in response.json()["detail"] or "Invalid file type" in response.json()["detail"]

def test_process_document_success():
    # We need a valid dummy PDF or Image. 
    # Create a small valid PDF bytes
    # Minimal PDF header
    pdf_content = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/MediaBox [0 0 595 842]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000111 00000 n\ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n190\n%%EOF'
    
    files = {'file': ('test_doc.pdf', pdf_content, 'application/pdf')}
    
    # Mocking background tasks is tricky with TestClient? 
    # TestClient runs fastAPI app synchronously, including background tasks.
    
    # Use unittest.mock
    from unittest.mock import patch
    
    # Patch SessionLocal used by background task to use our test DB
    with patch('main.processor.process_document') as mock_process, \
         patch('main.SessionLocal', return_value=TestingSessionLocal()):
        
        mock_process.return_value = {
            "status": "success", 
            "extracted_fields": {"test": "val"},
            "confidence": 0.9,
            "document_type": "invoice"
        }
        
        response = client.post("/api/v1/process", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        doc_id = data["data"]["document_id"]
        
        # Verify DB Entry exists
        # We need to access the DB used by the app. Since we use in-memory and override, 
        # we can open a new session from TestingSessionLocal to check state.
        
        db = TestingSessionLocal()
        from src.database.db import CRUD
        saved_doc = CRUD.get_document(db, doc_id)
        db.close()
        
        assert saved_doc is not None
        
        # TestClient runs background tasks synchronously, so it should be completed immediately after request returns
        assert saved_doc['status'] == 'completed'
        assert saved_doc['result_json']['extracted_fields']['test'][0]['value'] == "val"

def test_get_result():
    # Manually insert a completed doc
    db = TestingSessionLocal()
    from src.database.db import CRUD
    CRUD.create_document(db, "test_id_123", "test.pdf")
    CRUD.update_document_result(db, "test_id_123", {}, {"email": "test@test.com"}, "")
    db.close()
    
    response = client.get("/api/v1/result/test_id_123")
    assert response.status_code == 200
    assert response.json()["data"]["extracted_data"]["email"][0]["value"] == "test@test.com"

def test_export_csv():
    db = TestingSessionLocal()
    from src.database.db import CRUD
    CRUD.create_document(db, "test_csv", "test.pdf")
    CRUD.update_document_result(db, "test_csv", {}, {"email": "csv@test.com"}, "")
    db.close()
    
    response = client.get("/api/v1/export/test_csv?format=csv")
    assert response.status_code == 200
    # FastAPI/Starlette adds charset
    assert "text/csv" in response.headers["content-type"]
    assert "csv@test.com" in response.text
