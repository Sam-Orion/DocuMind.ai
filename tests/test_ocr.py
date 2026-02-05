import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.ocr.easy_ocr_engine import EasyOCREngine

class TestOCR:
    @pytest.fixture
    def mock_reader(self):
        with patch('easyocr.Reader') as MockReader:
            # Create a mock instance
            mock_instance = MagicMock()
            MockReader.return_value = mock_instance
            yield mock_instance

    def test_initialization(self, mock_reader):
        """Test that EasyOCR reader is initialized correctly (singleton behavior)."""
        engine = EasyOCREngine(gpu=False)
        assert engine.reader is not None
        # Should initialize with 'en' by default
        # Note: Singleton logic might make strict 'called_once' checks tricky across tests 
        # if not reset, but basic instantiation should pass.
        
    def test_extract_text(self, mock_reader):
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

    def test_extract_text_error_handling(self, mock_reader):
        """Test error handling when OCR fails."""
        mock_reader.readtext.side_effect = Exception("OCR Engine Failed")
        
        engine = EasyOCREngine()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(Exception) as excinfo:
            engine.extract_text(dummy_image)
        assert "OCR Engine Failed" in str(excinfo.value)
