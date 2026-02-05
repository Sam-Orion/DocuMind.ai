import os
import sys
import logging
import pytest
from pathlib import Path

# Add src to python path to verify imports works correctly from tests
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.image_processor import process
from src.ocr.easy_ocr_engine import EasyOCREngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ocr_pipeline():
    """
    Test the full OCR pipeline:
    1. Preprocessing (loading, enhancing, resizing)
    2. Text extraction using EasyOCR
    """
    # Path to sample invoice
    base_dir = Path(__file__).parent.parent
    sample_image_path = base_dir / "data" / "sample_documents" / "invoice.png"
    
    assert sample_image_path.exists(), f"Sample image not found at {sample_image_path}"
    
    logger.info(f"Testing OCR with image: {sample_image_path}")
    
    # 1. Preprocessing
    preprocessed_images = process(str(sample_image_path))
    assert len(preprocessed_images) > 0, "Preprocessing returned no images"
    
    processed_img = preprocessed_images[0]
    assert processed_img is not None, "Processed image is None"
    # Check if image is a numpy array
    assert hasattr(processed_img, 'shape'), "Processed image is not a numpy array"
    
    # 2. OCR Extraction
    # Initialize engine (en)
    ocr_engine = EasyOCREngine(language='en')
    results = ocr_engine.extract_text(processed_img)
    
    assert isinstance(results, list), "OCR results should be a list"
    assert len(results) > 0, "OCR returned no text results"
    
    # Log valid output
    logger.info("OCR Results sample:")
    for i, res in enumerate(results[:5]): # Print first 5 results
        logger.info(f"{i}: {res['text']} (Conf: {res['confidence']:.2f})")
        
    # Validation of structure
    first_result = results[0]
    assert 'text' in first_result
    assert 'bbox' in first_result
    assert 'confidence' in first_result
    
    # Verify confidence score is reasonable
    assert 0.0 <= first_result['confidence'] <= 1.0
    
    # Calculate average confidence
    avg_conf = sum(r['confidence'] for r in results) / len(results)
    logger.info(f"Average Confidence: {avg_conf:.2f}")
    
    assert avg_conf > 0.1, "Average confidence is too low, something might be wrong with the image or OCR"

if __name__ == "__main__":
    test_ocr_pipeline()
    print("Test passed!")
