import unittest
import sys
import logging
import numpy as np
import cv2
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock missing dependencies
sys.modules['pdf2image'] = MagicMock()
sys.modules['easyocr'] = MagicMock()
# Also mock pytest in case it's used in imports (though I removed it from my thought, let's be safe if I import the original test file)
sys.modules['pytest'] = MagicMock()

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.image_processor import process, load_document, enhance_image, resize_image
from src.ocr.easy_ocr_engine import EasyOCREngine

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestOCRPipeline(unittest.TestCase):
    def setUp(self):
        # Create a dummy image (black text on white background)
        self.image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(self.image, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        self.image_path = "dummy_image.png"
        cv2.imwrite(self.image_path, self.image)

    def tearDown(self):
        if os.path.exists(self.image_path):
            os.remove(self.image_path)

    def test_enhance_image(self):
        # This uses real OpenCV logic
        enhanced = enhance_image(self.image)
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, self.image.shape)
        # Check if it's still a numpy array
        self.assertTrue(isinstance(enhanced, np.ndarray))

    def test_resize_image(self):
        # Test resizing logic
        small_image = np.zeros((100, 100, 3), dtype=np.uint8)
        resized = resize_image(small_image)
        # Should be resized up to target width 1500
        self.assertEqual(resized.shape[1], 1500)
        
        large_image = np.zeros((2000, 2000, 3), dtype=np.uint8)
        resized_large = resize_image(large_image)
        # Should remain same if larger (logic: if w < target_width)
        # Wait, my logic was: if w < target_width: resize.
        # So 2000 > 1500, should not change.
        self.assertEqual(resized_large.shape[1], 2000)

    def test_load_document_image(self):
        # Test loading the created dummy image (uses real cv2.imread)
        loaded = load_document(self.image_path)
        self.assertEqual(len(loaded), 1)
        self.assertTrue(np.array_equal(loaded[0], self.image))

    @patch('src.preprocessing.image_processor.convert_from_path')
    def test_load_document_pdf(self, mock_convert):
        # Mock pdf return
        mock_pil_image = MagicMock()
        mock_pil_image.__array__ = MagicMock(return_value=self.image) # converting to np array
        # PIL to np array on a mock needs care. 
        # Actually src code: np.array(pil_img)
        # So mocking __array__ might work or side_effect.
        
        # Simpler: mock convert_from_path to return a real PIL image
        from PIL import Image
        real_pil = Image.fromarray(self.image)
        mock_convert.return_value = [real_pil]
        
        loaded = load_document("test_doc.pdf")
        self.assertEqual(len(loaded), 1)
        # Note: load_document converts RGB to BGR.
        # self.image is created with (0,0,0) (black) which is same in RGB/BGR.
        # White (255,255,255) is also same.
        self.assertEqual(loaded[0].shape, self.image.shape)

    def test_easy_ocr_engine(self):
        # Test engine initialization and extraction
        engine = EasyOCREngine()
        
        # Mock the reader's readtext method
        engine.reader.readtext.return_value = [
            ([[10, 10], [100, 10], [100, 50], [10, 50]], "TEST", 0.99)
        ]
        
        results = engine.extract_text(self.image)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['text'], "TEST")
        self.assertEqual(results[0]['confidence'], 0.99)
        self.assertEqual(results[0]['bbox'], [[10, 10], [100, 10], [100, 50], [10, 50]])

import os
if __name__ == '__main__':
    unittest.main()
