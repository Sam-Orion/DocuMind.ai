import logging
import time
from typing import Dict, Any, Union, List
import numpy as np

from src.preprocessing.image_processor import ImageProcessor
from src.ocr.tesseract_engine import TesseractEngine
from src.classification.rule_based import RuleBasedClassifier
from src.extraction.regex_extractor import RegexExtractor
from src.extraction.document_specific.invoice_extractor import InvoiceExtractor
from src.extraction.document_specific.receipt_extractor import ReceiptExtractor
from src.extraction.document_specific.resume_extractor import ResumeExtractor
from src.validation.validators import CrossFieldValidator
from src.validation.auto_correct import AutoCorrector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Orchestrates the document processing pipeline:
    Preprocessing -> OCR -> Classification -> Extraction -> Validation/Correction
    """

    def __init__(self):
        """
        Initialize all pipeline components.
        """
        logger.info("Initializing DocumentPipeline components...")
        self.image_processor = ImageProcessor()
        self.ocr_engine = TesseractEngine()
        self.classifier = RuleBasedClassifier()
        
        # Extractors
        self.regex_extractor = RegexExtractor()
        self.invoice_extractor = InvoiceExtractor()
        self.receipt_extractor = ReceiptExtractor()
        self.resume_extractor = ResumeExtractor()
        
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
            doc_type = classification_result['document_type']
            timings['classification'] = time.time() - start

            # 4. Extraction
            start = time.time()
            logger.info(f"Stage 4: Extraction (Type: {doc_type})...")
            
            if doc_type == 'invoice':
                extraction_results = self.invoice_extractor.extract(full_text)
            elif doc_type == 'receipt':
                extraction_results = self.receipt_extractor.extract(full_text)
            elif doc_type == 'resume':
                extraction_results = self.resume_extractor.extract(full_text)
            else:
                # Fallback to generic regex extraction
                if hasattr(self.regex_extractor, 'extract_all'):
                     extraction_results = self.regex_extractor.extract_all(full_text)
                else:
                    extraction_results = {
                        "email": self.regex_extractor.extract_emails(full_text),
                        "phone_number": self.regex_extractor.extract_phone_numbers(full_text),
                        "dates": self.regex_extractor.extract_dates(full_text),
                        "amounts": self.regex_extractor.extract_amounts(full_text),
                        "urls": self.regex_extractor.extract_urls(full_text)
                    }
                    
            timings['extraction'] = time.time() - start
            
            # 5. Validation & Auto-Correction
            start = time.time()
            logger.info("Stage 5: Validation & Correction...")
            
            # Apply corrections to known field types
            self._apply_corrections(extraction_results, doc_type)
            
            # Validate
            validation_report = CrossFieldValidator.validate(extraction_results, doc_type)
            
            timings['validation'] = time.time() - start

            total_duration = time.time() - pipeline_start
            
            logger.info(f"Pipeline completed in {total_duration:.2f} seconds.")

            return {
                "status": "success",
                "document_type": doc_type,
                "confidence": classification_result['confidence'],
                "extracted_fields": extraction_results,
                "validation_report": validation_report,
                "text_content": full_text,
                "ocr_details": ocr_result['details'],
                "classification_details": classification_result,
                "performance": {
                    "total_time": total_duration,
                    "breakdown": timings
                }
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "performance": {
                    "total_time": time.time() - pipeline_start,
                    "breakdown": timings
                }
            }

    def _apply_corrections(self, data: Dict[str, Any], doc_type: str):
        """
        Iterate through extracted fields and try to auto-correct standard formats.
        Modifies data in-place.
        """
        # Define fields to check based on typical extractor outputs
        # This is a bit manual, but we can look for "value" keys and corresponding field info
        
        # Generic recursive search or key-based?
        # Extractors usually return flat dicts or nested dicts where leaf nodes are Field objects
        
        for key, field in data.items():
            if isinstance(field, dict) and "value" in field:
                val = field["value"]
                
                # Correction logic
                new_val = None
                
                # Check key naming convention or field type if we had it
                # For now, approximate by key name
                if "date" in key.lower():
                    new_val = AutoCorrector.correct_date_format(str(val))
                elif "amount" in key.lower() or "total" in key.lower() or "price" in key.lower() or "cost" in key.lower() or "tax" in key.lower():
                    new_val = AutoCorrector.correct_amount_format(str(val))
                elif "phone" in key.lower():
                     corrected = AutoCorrector.correct_phone_format(str(val))
                     if corrected: new_val = corrected

                if new_val is not None and new_val != val:
                    field["original_value"] = val
                    field["value"] = new_val
                    field["corrected"] = True
                    logger.debug(f"Auto-corrected {key}: {val} -> {new_val}")

