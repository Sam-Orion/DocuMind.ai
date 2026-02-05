import pytest
from src.classification.rule_based import RuleBasedClassifier

@pytest.fixture
def classifier():
    return RuleBasedClassifier()

class TestClassification:
    def test_classify_invoice(self, classifier):
        text = """
        INVOICE # 12345
        Bill To: John Doe
        Date: 2023-01-01
        Subtotal: $100.00
        Tax: $10.00
        Total: $110.00
        """
        result = classifier.classify(text)
        assert result['document_type'] == "Invoice"
        assert result['confidence'] > 0.5
        assert "invoice" in [m.lower() for m in result['matched_rules']]

    def test_classify_resume(self, classifier):
        text = """
        John Doe
        Software Engineer
        
        Experience:
        - Senior Developer at Tech Corp
        
        Education:
        - BS Computer Science
        
        Skills: Python, AWS, Docker
        """
        result = classifier.classify(text)
        assert result['document_type'] == "Resume"
        assert result['confidence'] > 0.6
        assert "experience" in [m.lower() for m in result['matched_rules']]

    def test_classify_receipt(self, classifier):
        # Short text, typical of receipts
        text = """
        Walmart
        Transaction #9999
        Total Amount: $45.20
        Thank you for shopping!
        """
        result = classifier.classify(text)
        assert result['document_type'] == "Receipt"
        # Receipts often have lower confidence if text is sparse, but "thank you" is a strong signal
        assert result['confidence'] > 0.3 

    def test_classify_id_document(self, classifier):
        text = """
        REPUBLIC OF STATE
        PASSPORT
        Date of Birth: 01 Jan 1990
        Sex: M
        Expiry Date: 01 Jan 2030
        """
        result = classifier.classify(text)
        assert result['document_type'] == "ID Document"
        assert "passport" in [m.lower() for m in result['matched_rules']]

    def test_empty_input(self, classifier):
        result = classifier.classify("")
        assert result['document_type'] == "Unknown"
        assert result['confidence'] == 0.0
