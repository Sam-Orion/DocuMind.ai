import logging
import time
from typing import Dict, Any, Union, List
import numpy as np

from src.preprocessing.image_processor import ImageProcessor
from src.ocr.easy_ocr_engine import EasyOCREngine
from src.classification.rule_based import RuleBasedClassifier
from src.extraction.regex_extractor import RegexExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Orchestrates the document processing pipeline:
    Preprocessing -> OCR -> Classification -> Extraction
    """

    def __init__(self):
        """
        Initialize all pipeline components.
        """
        logger.info("Initializing DocumentPipeline components...")
        self.image_processor = ImageProcessor()
        self.ocr_engine = EasyOCREngine()  # Singleton handles repeated init
        self.classifier = RuleBasedClassifier()
        self.extractor = RegexExtractor()
        logger.info("DocumentPipeline initialized successfully.")

    def process_document(self, file_path_or_bytes: Union[str, bytes]) -> Dict[str, Any]:
        """
        Process a single document through the pipeline.

        Args:
            file_path_or_bytes: Path to image/PDF or raw bytes.

        Returns:
            Dict containing results from all stages and performance metrics.
        """
        pipeline_start = time.time()
        timings = {}
        
        try:
            # 1. Load & Preprocess
            start = time.time()
            logger.info("Stage 1: Preprocessing...")
            
            # TODO: Handle PDF vs Image input logic more robustly here if needed
            # For now, assuming load_image handles single images. 
            # If PDF, we might process just the first page or need a loop.
            # To keep it simple for now, we assume single image or first page of PDF conversion.
            
            if isinstance(file_path_or_bytes, str) and file_path_or_bytes.lower().endswith('.pdf'):
                # Simple handling: convert and take first page for MVP
                images = self.image_processor.convert_pdf_to_images(file_path_or_bytes)
                original_image = images[0]
            else:
                original_image = self.image_processor.load_image(file_path_or_bytes)

            # Resize for optimal OCR then Enhance
            resized_image = self.image_processor.resize_for_ocr(original_image)
            processed_image = self.image_processor.enhance_image(resized_image)
            
            timings['preprocessing'] = time.time() - start

            # 2. OCR
            start = time.time()
            logger.info("Stage 2: OCR...")
            ocr_result = self.ocr_engine.extract_text(processed_image)
            full_text = ocr_result['text']
            timings['ocr'] = time.time() - start

            # 3. Classification
            start = time.time()
            logger.info("Stage 3: Classification...")
            classification_result = self.classifier.classify(full_text)
            timings['classification'] = time.time() - start

            # 4. Extraction
            start = time.time()
            logger.info("Stage 4: Extraction...")
            extraction_results = {
                "emails": self.extractor.extract_emails(full_text),
                "phones": self.extractor.extract_phone_numbers(full_text),
                "dates": self.extractor.extract_dates(full_text),
                "amounts": self.extractor.extract_amounts(full_text),
                "invoice_numbers": self.extractor.extract_invoice_number(full_text),
                "urls": self.extractor.extract_urls(full_text)
            }
            timings['extraction'] = time.time() - start

            total_duration = time.time() - pipeline_start
            
            logger.info(f"Pipeline completed in {total_duration:.2f} seconds.")

            return {
                "status": "success",
                "document_type": classification_result['document_type'],
                "confidence": classification_result['confidence'],
                "extracted_fields": extraction_results,
                "text_content": full_text,
                "ocr_details": ocr_result['details'],
                "classification_details": classification_result,
                "performance": {
                    "total_time": total_duration,
                    "breakdown": timings
                }
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "performance": {
                    "total_time": time.time() - pipeline_start,
                    "breakdown": timings
                }
            }
