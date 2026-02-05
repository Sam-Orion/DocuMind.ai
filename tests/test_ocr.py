import pytest
import numpy as np
import sys
from unittest.mock import MagicMock, patch

# Mock pytesseract BEFORE importing TesseractEngine
# This ensures tests pass even if the library isn't installed in the env
mock_pytesseract_module = MagicMock()
sys.modules['pytesseract'] = mock_pytesseract_module

from src.ocr.tesseract_engine import TesseractEngine

class TestOCR:
    
    def setup_method(self):
        # Reset mocks before each test
        mock_pytesseract_module.reset_mock()

    def test_extract_text_success(self):
        """Test text extraction wrapper for Tesseract with valid output."""
        
        # Setup the mock behavior for this test
        mock_pytesseract_module.image_to_string.return_value = "Hello World"
        
        # Mock image_to_data (returns dict)
        mock_pytesseract_module.image_to_data.return_value = {
            'text': ['Hello', 'World', ''],
            'conf': [99, 95, -1],
            'left': [10, 60, 0],
            'top': [10, 10, 0],
            'width': [40, 40, 0],
            'height': [20, 20, 0]
        }
        # Tesseract Output enum mocking
        mock_pytesseract_module.Output.DICT = 'dict'

        engine = TesseractEngine()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = engine.extract_text(dummy_image)
        
        assert result['text'] == "Hello World"
        assert len(result['details']) == 2
        # Verify normalization of bounding box and confidence
        assert result['details'][0]['text'] == "Hello"
        assert result['details'][0]['bbox']['x'] == 10
        assert result['details'][0]['confidence'] == 0.99
        
        # Verify pytesseract was called
        mock_pytesseract_module.image_to_string.assert_called()

    def test_handling_missing_library(self):
        """Test error handling when pytesseract raises ImportError (simulated)."""
        # We simulate checking for the library by making the mock raise an error roughly
        # equivalent to what might happen, or we check our wrapper handles errors.
        
        mock_pytesseract_module.image_to_string.side_effect = ImportError("No module named pytesseract")
        
        engine = TesseractEngine()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # The engine catches generic exceptions or propagates them. 
        # In tesseract_engine.py we catch specific errors or generic ones.
        with pytest.raises(ImportError):
            engine.extract_text(dummy_image)

    def test_handling_missing_binary(self):
        """Test error handling when Tesseract binary is missing."""
        # Create a specific exception class for TesseractNotFoundError since we mocked the module
        class TesseractNotFoundError(Exception): pass
        mock_pytesseract_module.TesseractNotFoundError = TesseractNotFoundError
        
        mock_pytesseract_module.image_to_string.side_effect = TesseractNotFoundError("tesseract is not installed")
        
        engine = TesseractEngine()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(TesseractNotFoundError):
            engine.extract_text(dummy_image)
