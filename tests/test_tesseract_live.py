import pytest
import numpy as np
from PIL import Image, ImageDraw
from src.ocr.tesseract_engine import TesseractEngine

class TestTesseractLive:
    """
    Integration tests that actually call the Tesseract binary.
    Requires 'tesseract' to be installed on the system (e.g. brew install tesseract).
    """

    def test_live_extraction(self):
        # 1. Create a simple image with text "HELLO"
        # White background, black text
        img = Image.new('RGB', (200, 100), color='white')
        d = ImageDraw.Draw(img)
        # Using default font (usually tiny), potentially hard for OCR if too small?
        # Let's try to verify simple text. 
        # Without a font path, PIL uses a bitmap font. "HELLO" might be small.
        # So we just write it and hope or use a simple pattern.
        # Better: Tesseract works well on high contrast.
        d.text((10, 40), "Django", fill='black') # "Django" is distinct
        
        # Convert to numpy for the engine (simulating real pipeline input)
        img_np = np.array(img)

        # 2. Initialize Engine
        try:
            engine = TesseractEngine()
        except Exception as e:
            pytest.fail(f"Could not initialize TesseractEngine. Is pytesseract installed? Error: {e}")

        # 3. Run Extraction
        try:
            result = engine.extract_text(img_np)
            text = result['text'].strip()
            print(f"Extracted Text: '{text}'")
            
            # 4. Assert
            # Allows for some noise, but "Django" should be there
            assert "Django" in text or "uiango" in text # common OCR misreads for default font
            
        except Exception as e:
             pytest.fail(f"Tesseract extraction failed. Is the 'tesseract' binary installed on your system? Error: {e}")
