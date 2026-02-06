import pytest
from src.validation.validators import FieldValidator, InvoiceValidator, CrossFieldValidator

class TestFieldValidator:
    
    def test_validate_email(self):
        assert FieldValidator.validate_email("test@example.org") is True
        assert FieldValidator.validate_email("invalid-email") is False
        assert FieldValidator.validate_email("user@example.com") is False # Blacklisted domain check in code
        
    def test_validate_phone(self):
        # US Number
        assert FieldValidator.validate_phone_number("555-123-4567") is False # Invalid by phonenumbers lib std strictly (needs area code validation usually)
        # Use a real looking fake num
        assert FieldValidator.validate_phone_number("+1 650 253 0000") is True
        assert FieldValidator.validate_phone_number("123") is False
        
    def test_validate_date(self):
        assert FieldValidator.validate_date("2023-01-01") is True
        assert FieldValidator.validate_date("1800-01-01") is False # Too old
        assert FieldValidator.validate_date("3000-01-01") is False # Too future
        assert FieldValidator.validate_date("invalid") is False
        
    def test_validate_amount(self):
        assert FieldValidator.validate_amount(100.50) is True
        assert FieldValidator.validate_amount("100.50") is True
        assert FieldValidator.validate_amount(-5.00) is False

class TestInvoiceValidator:
    
    def test_validate_totals_correct(self):
        data = {
            "subtotal": {"value": 100.00},
            "tax": {"value": 10.00},
            "shipping": {"value": 5.00},
            "discount": {"value": 0.00},
            "total_amount": {"value": 115.00}
        }
        res = InvoiceValidator.validate_totals(data)
        assert res["valid"] is True
        
    def test_validate_totals_incorrect(self):
        data = {
            "subtotal": {"value": 100.00},
            "tax": {"value": 10.00},
            "total_amount": {"value": 150.00} # Should be 110
        }
        res = InvoiceValidator.validate_totals(data)
        assert res["valid"] is False
        assert "Total mismatch" in res["errors"][0]

    def test_validate_dates(self):
        data = {
            "invoice_date": {"value": "2023-01-01"},
            "due_date": {"value": "2023-01-31"}
        }
        assert InvoiceValidator.validate_dates(data)["valid"] is True
        
        data["due_date"]["value"] = "2022-12-31"
        assert InvoiceValidator.validate_dates(data)["valid"] is False

class TestCrossFieldValidator:
    
    def test_validate_invoice(self):
        data = {
            "subtotal": {"value": "100.00"},
            "tax": {"value": "10.00"},
            "total_amount": {"value": "110.00"},
            "invoice_date": {"value": "2023-01-01"},
            "due_date": {"value": "2023-02-01"}
        }
        report = CrossFieldValidator.validate(data, "invoice")
        assert report["document_valid"] is True
        assert report["field_validations"]["total_amount"] is True
        
    def test_validate_invoice_invalid(self):
        data = {
            "subtotal": {"value": "100.00"},
            "tax": {"value": "10.00"},
            "total_amount": {"value": "999.00"} # Invalid Logic
        }
        report = CrossFieldValidator.validate(data, "invoice")
        assert report["document_valid"] is False
        assert len(report["logic_validations"]) > 0
