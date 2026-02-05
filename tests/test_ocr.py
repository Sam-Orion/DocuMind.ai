import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# This import will fail if dependencies (easyocr, python-bidi) are missing
import easyocr 
from src.ocr.easy_ocr_engine import EasyOCREngine

class TestOCR:
    def setup_method(self):
        # Reset the singleton for each test
        EasyOCREngine._reader_instance = None

    @patch('src.ocr.easy_ocr_engine.easyocr.Reader') 
    def test_initialization(self, mock_reader_cls):
        """
        Test that EasyOCR reader is initialized correctly.
        We mock the Reader class to avoid downloading models.
        """
        mock_instance = MagicMock()
        mock_reader_cls.return_value = mock_instance
        
        engine = EasyOCREngine(gpu=False)
        assert engine.reader is not None
        mock_reader_cls.assert_called_once()

    @patch('src.ocr.easy_ocr_engine.easyocr.Reader')
    def test_extract_text(self, mock_reader_cls):
        """Test text extraction wrapper."""
        # Setup mock behavior
        mock_instance = MagicMock()
        mock_reader_cls.return_value = mock_instance
        
        # Format: (bbox, text, prob)
        mock_bbox = [[10, 10], [100, 10], [100, 50], [10, 50]]
        mock_instance.readtext.return_value = [
            (mock_bbox, "Hello", 0.99),
            (mock_bbox, "World", 0.95)
        ]

        engine = EasyOCREngine()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result = engine.extract_text(dummy_image)
        
        assert result['text'] == "Hello\nWorld"
        assert len(result['details']) == 2
        assert result['details'][0]['text'] == "Hello"

    @patch('src.ocr.easy_ocr_engine.easyocr.Reader')
    def test_extract_text_error_handling(self, mock_reader_cls):
        """Test error handling when OCR fails."""
        mock_instance = MagicMock()
        mock_reader_cls.return_value = mock_instance
        mock_instance.readtext.side_effect = Exception("OCR Engine Failed")
        
        engine = EasyOCREngine()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(Exception) as excinfo:
            engine.extract_text(dummy_image)
        assert "OCR Engine Failed" in str(excinfo.value)
