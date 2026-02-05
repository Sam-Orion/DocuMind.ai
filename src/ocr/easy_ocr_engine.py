import logging
import easyocr
import numpy as np
from typing import List, Dict, Union, Any
import time
import ssl

# Workaround for SSL certificate verification failure on some systems during download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EasyOCREngine:
    def __init__(self, language: str = 'en'):
        """
        Initialize the EasyOCR reader.
        
        Args:
            language (str): Language code for OCR (default: 'en').
        """
        try:
            logger.info(f"Initializing EasyOCR for language: {language}")
            
            # Determine local model storage to avoid permission issues in ~/.EasyOCR
            # Assuming structure is src/ocr/easy_ocr_engine.py -> up 2 levels -> root -> models
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            model_dir = os.path.join(project_root, 'models')
            
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                
            logger.info(f"Using model storage directory: {model_dir}")

            # gpu=False by default to ensure compatibility, can be made configurable
            self.reader = easyocr.Reader(
                [language], 
                gpu=True,
                model_storage_directory=model_dir,
                user_network_directory=model_dir, # safe to use same dir
                download_enabled=True
            ) 
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            # Fallback to CPU if GPU fails (though EasyOCR usually handles this, specific errors might occur)
            try:
                logger.warning("Retrying EasyOCR initialization on CPU...")
                # Recalculate model dir variable as it's local scope above (though strictly it's available, let's keep it safe)
                # Actually simpler to just reference self.reader creation again with cpu
                self.reader = easyocr.Reader(
                    [language], 
                    gpu=False,
                    model_storage_directory=model_dir,
                    user_network_directory=model_dir,
                    download_enabled=True
                )
            except Exception as e2:
                logger.critical(f"Critical failure initializing EasyOCR: {e2}")
                raise

    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text from a preprocessed image using EasyOCR.
        
        Args:
            image (np.ndarray): Input image (numpy array, BGR or RGB).
            
        Returns:
            List[Dict[str, Any]]: List of extracted text segments with metadata.
                Each item contains:
                - 'text': The extracted text string.
                - 'bbox': Bounding box coordinates [[x1, y1], [x2, y1], [x2, y2], [x1, y2]].
                - 'confidence': Confidence score (0.0 to 1.0).
        """
        results = []
        try:
            start_time = time.time()
            # EasyOCR readtext expects image path, numpy array (RGB or BGR), or bytes
            # detail=1 returns (bbox, text, prob)
            raw_results = self.reader.readtext(image, detail=1)
            processing_time = time.time() - start_time
            logger.info(f"OCR processing finished in {processing_time:.4f} seconds. Found {len(raw_results)} text segments.")

            for (bbox, text, prob) in raw_results:
                # Convert bbox points to standard Python types (often they are numpy types)
                clean_bbox = [[int(pt[0]), int(pt[1])] for pt in bbox]
                
                results.append({
                    'text': text,
                    'bbox': clean_bbox,
                    'confidence': float(prob)
                })
                
            return results
        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
            return []
