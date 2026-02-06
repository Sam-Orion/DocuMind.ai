import pytest
import os
from src.extraction.document_specific.invoice_extractor import InvoiceExtractor

# Path to the sample file we just created
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "samples", "sample_invoice.txt")

@pytest.fixture
def invoice_text():
    with open(SAMPLE_PATH, "r") as f:
        return f.read()

@pytest.fixture
def extractor():
    return InvoiceExtractor()

class TestInvoiceExtractor:
    
    def test_extract_invoice_header(self, extractor, invoice_text):
        header = extractor.extract_invoice_header(invoice_text)
        assert header["invoice_number"]["value"] == "INV-2023-1001"
        assert header["invoice_date"]["value"] == "2023-11-01"
        assert "PO-998877" in header["po_number"]["value"]

    def test_extract_line_items(self, extractor, invoice_text):
        items = extractor.extract_line_items(invoice_text)
        assert len(items) == 3
        
        # Check first item
        assert items[0]["description"]["value"] == "Consulting Services"
        assert items[0]["quantity"]["value"] == "10"
        assert items[0]["total_amount"]["value"] == "1500.00"
        
        # Check last item
        assert items[2]["description"]["value"] == "Training Materials"
        assert items[2]["total_amount"]["value"] == "250.00"

    def test_extract_totals(self, extractor, invoice_text):
        # We need line items for validation logic, although extracting totals mostly relies on text
        line_items = extractor.extract_line_items(invoice_text)
        totals = extractor.extract_totals(invoice_text, line_items)
        
        assert totals["subtotal"]["value"] == 2250.00
        assert totals["tax"]["value"] == 225.00
        assert totals["total_amount"]["value"] == 2475.00
        assert len(totals["validation_errors"]) == 0

    def test_extract_payment_terms(self, extractor, invoice_text):
        terms = extractor.extract_payment_terms(invoice_text)
        assert "Net 15" in terms["term_description"]["value"] or "15" in terms["term_description"]["value"]

    def test_full_extraction_structure(self, extractor, invoice_text):
        result = extractor.extract(invoice_text)
        assert result["document_type"] == "invoice"
        assert "header" in result
        assert "line_items" in result
        assert "totals" in result
        assert len(result["line_items"]) == 3
