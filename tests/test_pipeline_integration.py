import pytest
from unittest.mock import MagicMock, patch
from src.pipeline import DocumentProcessor

class TestPipelineIntegration:
    
    @pytest.fixture
    def processor(self):
        return DocumentProcessor()
        
    @patch('src.pipeline.ImageProcessor')
    @patch('src.pipeline.TesseractEngine')
    @patch('src.pipeline.RuleBasedClassifier')
    @patch('src.pipeline.InvoiceExtractor')
    def test_invoice_routing_and_validation(self, MockInvoiceExtractor, MockClassifier, MockOcr, MockImageProcessor):
        
        # Setup mocks
        processor = DocumentProcessor()
        
        # 1. OCR Returns invoice text
        processor.ocr_engine.extract_text.return_value = {
            'text': "Invoice #123\nDate: Jan 5, 2023\nTotal: $100.00",
            'details': []
        }
        
        # 2. Classifier identifies it as invoice
        processor.classifier.classify.return_value = {
            'document_type': 'invoice',
            'confidence': 0.95
        }
        
        # 3. Invoice Extractor returns fields
        # Note: We simulate a field that needs correction (date format)
        processor.invoice_extractor.extract.return_value = {
            "invoice_number": {"value": "123", "confidence": 0.9},
            "invoice_date": {"value": "Jan 5, 2023", "confidence": 0.9}, # Needs correction
            "total_amount": {"value": "1OO.00", "confidence": 0.8}, # Needs correction (OCR error)
            "subtotal": {"value": "90.00", "confidence": 0.8},
            "tax": {"value": "10.00", "confidence": 0.8}
        }
        
        # Run pipeline
        result = processor.process_document("dummy.jpg")
        
        # Assertions
        assert result['status'] == 'success'
        assert result['document_type'] == 'invoice'
        
        # Check if routing worked
        processor.invoice_extractor.extract.assert_called_once()
        
        # Check Auto-Correction
        # Date should be ISO
        date_field = result['extracted_fields']['invoice_date']
        assert date_field['value'] == "2023-01-05"
        assert date_field.get('corrected') is True
        
        # Total amount should be float 100.00 (from 1OO.00)
        total_field = result['extracted_fields']['total_amount']
        assert total_field['value'] == 100.0
        assert total_field.get('corrected') is True
        
        # Check Validation Report
        report = result['validation_report']
        assert report is not None
        # Should be valid because corrected values 90+10=100 match
        # CrossFieldValidator checks total_amount against subtotal+tax
        # 90+10 = 100. Corrected total is 100. So it should be valid.
        assert report['document_valid'] is True # Assuming logic checks match

    @patch('src.pipeline.ImageProcessor')
    @patch('src.pipeline.TesseractEngine')
    @patch('src.pipeline.RuleBasedClassifier')
    def test_fallback_routing(self, MockClassifier, MockOcr, MockImageProcessor):
        processor = DocumentProcessor()
        
        processor.ocr_engine.extract_text.return_value = {'text': "Some random text", 'details': []}
        processor.classifier.classify.return_value = {'document_type': 'unknown', 'confidence': 0.5}
        
        result = processor.process_document("dummy.jpg")
        
        assert result['document_type'] == 'unknown'
        # Should contain generic regex fields
        assert 'email' in result['extracted_fields']
