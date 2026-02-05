import logging
import time
import numpy as np
from typing import List, Dict, Any, Union, Tuple
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EasyOCREngine:
    """
    Wrapper around EasyOCR to handle text extraction with performance monitoring and caching.
    """
    
    _reader_instance = None

    def __init__(self, languages: List[str] = ['en'], gpu: bool = False):
        """
        Initialize the EasyOCR engine.
        Uses a singleton pattern for the reader to avoid reloading models.
        """
        # Lazy import to allow class usage even if easyocr is not installed (for testing/dev)
        try:
            import easyocr
        except ImportError:
            logger.error("EasyOCR not installed or broken. OCR will fail.")
            easyocr = None

        if EasyOCREngine._reader_instance is None and easyocr:
            logger.info(f"Initializing EasyOCR reader for languages: {languages}, GPU={gpu}...")
            start_time = time.time()
            try:
                EasyOCREngine._reader_instance = easyocr.Reader(languages, gpu=gpu)
                end_time = time.time()
                logger.info(f"EasyOCR initialized in {end_time - start_time:.2f} seconds.")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {str(e)}")
                raise e
        elif easyocr:
            logger.info("Using cached EasyOCR reader instance.")

        self.reader = EasyOCREngine._reader_instance

    def extract_text(self, image: Union[str, np.ndarray, bytes]) -> Dict[str, Any]:
        """
        Extract text from an image using EasyOCR.

        Args:
            image: Image path, numpy array, or bytes.

        Returns:
            Dict containing:
            - 'text': Full extracted text as a single string.
            - 'details': List of dicts with 'text', 'bbox', 'confidence'.
            - 'processing_time': Time taken for OCR in seconds.
        """
        start_time = time.time()
        logger.info("Starting OCR extraction...")

        try:
            # EasyOCR handles file paths, numpy arrays, and bytes directly
            results = self.reader.readtext(image)
            
            extracted_data = []
            full_text_parts = []

            for (bbox, text, prob) in results:
                # bbox is a list of 4 points [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                # Convert numpy types to native python types for JSON serialization
                clean_bbox = [[int(p[0]), int(p[1])] for p in bbox]
                
                item = {
                    "text": text,
                    "bbox": clean_bbox,
                    "confidence": float(prob)
                }
                extracted_data.append(item)
                full_text_parts.append(text)

            full_text = "\n".join(full_text_parts)
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"OCR completed in {duration:.2f} seconds. Extracted {len(extracted_data)} text blocks.")

            return {
                "text": full_text,
                "details": extracted_data,
                "processing_time": duration
            }

        except Exception as e:
            logger.error(f"Error during OCR extraction: {str(e)}")
            raise e
