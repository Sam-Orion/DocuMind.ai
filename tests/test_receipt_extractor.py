import pytest
import os
from src.extraction.document_specific.receipt_extractor import ReceiptExtractor

# Path to the sample file we just created
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "samples", "sample_receipt.txt")

@pytest.fixture
def receipt_text():
    with open(SAMPLE_PATH, "r") as f:
        return f.read()

@pytest.fixture
def extractor():
    return ReceiptExtractor()

class TestReceiptExtractor:
    
    def test_extract_merchant_info(self, extractor, receipt_text):
        info = extractor.extract_merchant_info(receipt_text)
        # Assuming heuristic picks up first line
        assert "SuperMart 123" in info["name"]["value"]
        assert "555" in info["phone"]["value"] 

    def test_extract_transaction_details(self, extractor, receipt_text):
        details = extractor.extract_transaction_details(receipt_text)
        assert details["date"]["value"] == "2023-11-05"
        assert "14:30" in details["time"]["value"]
        assert "9876543210" in details["receipt_number"]["value"]
        assert details["terminal_id"]["value"] == "5"

    def test_extract_items(self, extractor, receipt_text):
        items = extractor.extract_items(receipt_text)
        assert len(items) == 4
        
        # Check specific items
        assert items[0]["description"]["value"] == "Organic Bananas"
        assert items[0]["total_price"]["value"] == "2.99"
        
        assert items[3]["description"]["value"] == "Dark Chocolate"
        assert items[3]["total_price"]["value"] == "5.99"

    def test_extract_payment_info(self, extractor, receipt_text):
        payment = extractor.extract_payment_info(receipt_text)
        assert "Visa" in payment["method"]["value"]
        assert "4321" in payment["card_last_4"]["value"]
        assert "1234AB" in payment["auth_code"]["value"]

    def test_extract_loyalty_info(self, extractor, receipt_text):
        loyalty = extractor.extract_loyalty_info(receipt_text)
        assert "888777666" in loyalty["member_id"]["value"]
        assert "540" in str(loyalty["points_balance"]["value"])

    def test_find_total(self, extractor, receipt_text):
        # We need items for _find_total signature, though it mostly parses text
        items = extractor.extract_items(receipt_text)
        # Access private-ish method via finding it in full extract or testing direct if needed
        # We'll test via full extract primarily, but let's assume it's exposed or wrapped
        # The class exposes 'extract' which calls it.
        result = extractor.extract(receipt_text)
        assert result["total_amount"]["value"] == 17.81
