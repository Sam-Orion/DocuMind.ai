import pytest
import sys
from unittest.mock import MagicMock, patch
import numpy as np

# Create a mock for easyocr before importing the engine
mock_easyocr = MagicMock()
mock_reader = MagicMock()
mock_easyocr.Reader.return_value = mock_reader
sys.modules['easyocr'] = mock_easyocr

# Now import the engine which will check sys.modules or try to import (and get our mock)
from src.ocr.easy_ocr_engine import EasyOCREngine

class TestOCR:
    def setup_method(self):
        # Reset the singleton and reader for each test
        EasyOCREngine._reader_instance = None
        mock_reader.reset_mock()

    def test_initialization(self):
        """Test that EasyOCR reader is initialized correctly (singleton behavior)."""
        engine = EasyOCREngine(gpu=False)
        assert engine.reader is not None
        mock_easyocr.Reader.assert_called()

    def test_extract_text(self):
        """Test text extraction wrapper."""
        # Setup mock return value for readtext
        # Format: (bbox, text, prob)
        mock_bbox = [[10, 10], [100, 10], [100, 50], [10, 50]]
        mock_reader.readtext.return_value = [
            (mock_bbox, "Hello", 0.99),
            (mock_bbox, "World", 0.95)
        ]

        engine = EasyOCREngine()
        # Create a dummy image (black square)
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = engine.extract_text(dummy_image)
        
        assert result['text'] == "Hello\nWorld"
        assert len(result['details']) == 2
        assert result['details'][0]['text'] == "Hello"
        assert result['details'][0]['confidence'] == 0.99
        assert 'processing_time' in result

    def test_extract_text_error_handling(self):
        """Test error handling when OCR fails."""
        mock_reader.readtext.side_effect = Exception("OCR Engine Failed")
        
        engine = EasyOCREngine()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(Exception) as excinfo:
            engine.extract_text(dummy_image)
        assert "OCR Engine Failed" in str(excinfo.value)
