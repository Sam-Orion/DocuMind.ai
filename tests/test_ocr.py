import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.ocr.tesseract_engine import TesseractEngine

class TestOCR:
    
    @patch('src.ocr.tesseract_engine.pytesseract')
    def test_extract_text_success(self, mock_pytesseract):
        """Test text extraction wrapper for Tesseract with valid output."""
        
        # Setup the mock behavior for this test
        mock_pytesseract.image_to_string.return_value = "Hello World"
        
        # Mock image_to_data (returns dict)
        mock_pytesseract.image_to_data.return_value = {
            'text': ['Hello', 'World', ''],
            'conf': [99, 95, -1],
            'left': [10, 60, 0],
            'top': [10, 10, 0],
            'width': [40, 40, 0],
            'height': [20, 20, 0]
        }
        # Tesseract Output enum mocking - need to ensure it's accessible
        mock_pytesseract.Output.DICT = 'dict'

        engine = TesseractEngine()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = engine.extract_text(dummy_image)
        
        assert result['text'] == "Hello World"
        assert len(result['details']) == 2
        # Verify normalization of bounding box and confidence
        assert result['details'][0]['text'] == "Hello"
        assert result['details'][0]['bbox']['x'] == 10
        assert result['details'][0]['confidence'] == 0.99
        
    @patch('src.ocr.tesseract_engine.pytesseract')
    def test_handling_missing_library(self, mock_pytesseract):
        """Test error handling when pytesseract raises ImportError."""
        
        mock_pytesseract.image_to_string.side_effect = ImportError("No module named pytesseract")
        
        engine = TesseractEngine()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ImportError):
            engine.extract_text(dummy_image)

    @patch('src.ocr.tesseract_engine.pytesseract')
    def test_handling_missing_binary(self, mock_pytesseract):
        """Test error handling when Tesseract binary is missing."""
        # Create a specific exception class for TesseractNotFoundError and attach to mock
        class TesseractNotFoundError(Exception): pass
        mock_pytesseract.TesseractNotFoundError = TesseractNotFoundError
        
        mock_pytesseract.image_to_string.side_effect = TesseractNotFoundError("tesseract is not installed")
        
        engine = TesseractEngine()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(TesseractNotFoundError):
            engine.extract_text(dummy_image)
