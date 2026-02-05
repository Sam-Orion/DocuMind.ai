import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.ocr.tesseract_engine import TesseractEngine

class TestOCR:
    
    @patch('src.ocr.tesseract_engine.pytesseract')
    def test_extract_text(self, mock_pytesseract):
        """Test text extraction wrapper for Tesseract."""
        
        # Mock image_to_string
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
        mock_pytesseract.Output.DICT = 'dict'

        engine = TesseractEngine()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = engine.extract_text(dummy_image)
        
        assert result['text'] == "Hello World"
        assert len(result['details']) == 2
        assert result['details'][0]['text'] == "Hello"
        assert result['details'][0]['bbox']['x'] == 10
        assert result['details'][0]['confidence'] == 0.99

    @patch('src.ocr.tesseract_engine.pytesseract')
    def test_tesseract_not_found(self, mock_pytesseract):
        """Test error when binary is missing."""
        mock_pytesseract.image_to_string.side_effect = Exception("Tesseract not found")
        # Note: In real life pytesseract raises TesseractNotFoundError, but regular Exception is fine for general error catch test
        
        engine = TesseractEngine()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(Exception):
            engine.extract_text(dummy_image)
