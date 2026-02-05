import pytest
from src.extraction.regex_extractor import RegexExtractor

@pytest.fixture
def extractor():
    return RegexExtractor()

class TestExtraction:
    def test_extract_emails(self, extractor):
        text = "Contact us at support@documind.ai or sales@example.com."
        results = extractor.extract_emails(text)
        assert len(results) == 2
        assert results[0]['value'] == "support@documind.ai"
        assert results[1]['value'] == "sales@example.com"
        assert results[0]['field_type'] == "email"

    def test_extract_dates(self, extractor):
        # Test various date formats
        text = "Invoice Date: 2023-10-25. Due: 15/11/2023. Paid on Dec 1, 2023."
        results = extractor.extract_dates(text)
        assert len(results) == 3
        # Normalized ISO output
        assert results[0]['value'] == "2023-10-25"
        assert results[1]['value'] == "2023-11-15" 
        assert results[2]['value'] == "2023-12-01"

    def test_extract_amounts(self, extractor):
        text = "Total: $1,250.00. Tax: USD 50.00. Discount: 10.00"
        results = extractor.extract_amounts(text)
        assert len(results) >= 2
        values = [r['value'] for r in results]
        assert 1250.00 in values
        assert 50.00 in values

    def test_extract_invoice_number(self, extractor):
        text = "Invoice Number: INV-2023-001"
        results = extractor.extract_invoice_number(text)
        assert len(results) == 1
        assert results[0]['value'] == "INV-2023-001"
        assert results[0]['confidence'] > 0.8

    def test_extract_urls(self, extractor):
        text = "Visit https://www.documind.ai for more info."
        results = extractor.extract_urls(text)
        assert len(results) == 1
        assert results[0]['value'] == "https://www.documind.ai"
