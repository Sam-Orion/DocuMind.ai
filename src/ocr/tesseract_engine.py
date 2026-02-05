import logging
import time
import pytesseract
from PIL import Image
import numpy as np
from typing import Dict, Any, Union, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TesseractEngine:
    """
    Wrapper around Tesseract OCR (via pytesseract) for text extraction.
    Requires Tesseract binary to be installed on the system.
    """

    def __init__(self, languages: str = 'eng'):
        """
        Initialize Tesseract Engine.
        
        Args:
            languages: Tesseract language code (default 'eng'). Multiple can be joined by '+' (e.g. 'eng+fra').
        """
        self.languages = languages
        logger.info(f"Initialized TesseractEngine with languages: {languages}")

    def extract_text(self, image: Union[str, np.ndarray, bytes, Image.Image]) -> Dict[str, Any]:
        """
        Extract text from an image using Tesseract.

        Args:
            image: Image source (path, numpy array, bytes, or PIL Image).

        Returns:
             Dict containing:
            - 'text': Full extracted text.
            - 'details': List of words with bbox and confidence.
            - 'processing_time': Duration in seconds.
        """
        start_time = time.time()
        logger.info("Starting Tesseract OCR extraction...")

        try:
            # Convert input to PIL Image if necessary, as pytesseract prefers it
            if isinstance(image, bytes):
                img_obj = Image.open(io.BytesIO(image))
            elif isinstance(image, np.ndarray):
                img_obj = Image.fromarray(image)
            elif isinstance(image, str):
                img_obj = Image.open(image)
            else:
                img_obj = image

            # 1. Extract Full Text
            full_text = pytesseract.image_to_string(img_obj, lang=self.languages)

            # 2. Extract Data with Confidence and Bounding Boxes
            # pytesseract.image_to_data returns a dict with lists of values
            data = pytesseract.image_to_data(img_obj, lang=self.languages, output_type=pytesseract.Output.DICT)
            
            extracted_data = []
            num_boxes = len(data['text'])
            
            for i in range(num_boxes):
                # Filter out empty text (often just structure/noise)
                if int(data['conf'][i]) > -1 and data['text'][i].strip():
                    item = {
                        "text": data['text'][i],
                        # Tesseract returns x, y, w, h. We convert to [[x,y], [x+w, y], [x+w, y+h], [x, y+h]] 
                        # to match the previous format if possible, or just keep simpler [x, y, w, h]
                        # Let's use [x, y, w, h] for standard tesseract usage usually, 
                        # but to maintain compatibility with our frontend which might expect points:
                        # documind-ai frontend (app.py) doesn't explicitly draw boxes yet, 
                        # but let's standardize on a simple box format.
                        "bbox": {
                            "x": data['left'][i], 
                            "y": data['top'][i], 
                            "w": data['width'][i], 
                            "h": data['height'][i]
                        },
                        "confidence": float(data['conf'][i]) / 100.0 # Tesseract is 0-100
                    }
                    extracted_data.append(item)

            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"OCR completed in {duration:.2f} seconds. Extracted {len(extracted_data)} words.")

            return {
                "text": full_text.strip(),
                "details": extracted_data,
                "processing_time": duration
            }

        except ImportError:
            logger.error("pytesseract library not found. Please pip install pytesseract.")
            raise
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract binary not found. Please install tesseract-ocr.")
            raise
        except Exception as e:
            logger.error(f"Error during Tesseract extraction: {str(e)}")
            raise e
