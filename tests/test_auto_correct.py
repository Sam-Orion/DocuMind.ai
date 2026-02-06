import pytest
from src.validation.auto_correct import AutoCorrector

class TestAutoCorrector:
    
    def test_correct_date_format(self):
        # Good formats
        assert AutoCorrector.correct_date_format("2023-01-01") == "2023-01-01"
        assert AutoCorrector.correct_date_format("Jan 5, 2023") == "2023-01-05"
        assert AutoCorrector.correct_date_format("01/05/2023") == "2023-01-05" # Assumes US format MDY by default in dateutil usually
        assert AutoCorrector.correct_date_format("5th January 2023") == "2023-01-05"
        
        # Fuzzy parsing
        assert AutoCorrector.correct_date_format("Due: Jan 5, 2023") == "2023-01-05"
        
        # Invalid
        assert AutoCorrector.correct_date_format("Not a date") is None

    def test_correct_amount_format(self):
        assert AutoCorrector.correct_amount_format("1,234.56") == 1234.56
        assert AutoCorrector.correct_amount_format("$1,234.56") == 1234.56
        assert AutoCorrector.correct_amount_format("$ 1 234.56") == 1234.56
        assert AutoCorrector.correct_amount_format("1.234,56") == 1234.56 # EU format inversion logic check
        assert AutoCorrector.correct_amount_format("100,50") == 100.50 # Comma decimal
        
        # Cleanup
        assert AutoCorrector.correct_amount_format("USD 50.00") == 50.00
        
        # Invalid
        assert AutoCorrector.correct_amount_format("Free") is None

    def test_correct_phone_format(self):
        # US
        assert AutoCorrector.correct_phone_format("555-123-4567") is None # Invalid area code usually (555) but technically structural check
        # Let's use a real-ish one
        assert AutoCorrector.correct_phone_format("650-253-0000") == "+16502530000"
        assert AutoCorrector.correct_phone_format("(650) 253-0000") == "+16502530000"
        assert AutoCorrector.correct_phone_format("650.253.0000") == "+16502530000"
        
        # Invalid
        assert AutoCorrector.correct_phone_format("12345") is None

    def test_suggest_corrections(self):
        # Amount OCR fix
        # "1OO.00" -> 100.00
        sug = AutoCorrector.suggest_corrections("1OO.OO", "amount", 0.5)
        assert 100.00 in sug
        
        sug = AutoCorrector.suggest_corrections("$l0.00", "currency", 0.5) # l -> 1
        assert 10.00 in sug
        
        # Date fix
        sug = AutoCorrector.suggest_corrections("Jan 5, 2023", "date", 0.5)
        assert "2023-01-05" in sug
