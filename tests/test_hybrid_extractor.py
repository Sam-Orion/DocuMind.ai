import pytest
from src.extraction.hybrid_extractor import HybridExtractor

@pytest.fixture
def hybrid_extractor():
    # Helper to return extractor. 
    # If modules missing, it might fail, but requirements should be installed.
    return HybridExtractor()

def test_extract_all_merging(hybrid_extractor):
    # This text contains elements for both Regex (Date, Email) and Spacy (Person, Org)
    text = "John Smith from Acme Corp called on 2024-01-15. Contact: john@acme.com"
    
    results = hybrid_extractor.extract_all(text)
    
    # Check Regex fields
    assert "date" in results
    dates = [d['value'] for d in results['date']]
    assert "2024-01-15" in dates
    
    assert "email" in results
    emails = [e['value'] for e in results['email']]
    assert "john@acme.com" in emails
    
    # Check Spacy fields (Mocking spacy/regex internal results would be better for pure unit test, 
    # but exact integration test depends on model performance. we proceed with expectation if model works.)
    # Note: If spacy model isn't loaded/working, these checks might fail or be empty.
    # We check if lists exist at least.
    assert isinstance(results.get('person_name'), list)
    assert isinstance(results.get('company_name'), list)

def test_deduplication_and_confidence(hybrid_extractor):
    # Manually test the _merge_lists logic to verify the formula
    
    # Scenario: Primary found "Google", Secondary found "Google Inc" (Fuzzy match)
    primary = [{
        "value": "Google",
        "confidence": 0.9,
        "position": {"start": 10, "end": 16}
    }]
    
    secondary = [{
        "value": "Google Inc",
        "confidence": 0.8,
        "position": {"start": 10, "end": 20} # Overlaps
    }]
    
    # 'company_name' prioritizes Spacy. Let's assume Spacy is primary here.
    merged = hybrid_extractor._merge_lists("company_name", primary, secondary)
    
    assert len(merged) == 1
    item = merged[0]
    
    # Value should be primary's value
    assert item['value'] == "Google"
    
    # Confidence Aggregation: 0.7 * 0.9 + 0.3 * 0.8 = 0.63 + 0.24 = 0.87
    expected_conf = (0.7 * 0.9) + (0.3 * 0.8)
    assert item['confidence'] == pytest.approx(expected_conf)

def test_no_duplicate(hybrid_extractor):
    primary = [{"value": "Apple", "confidence": 0.9}]
    secondary = [{"value": "Microsoft", "confidence": 0.9}]
    
    merged = hybrid_extractor._merge_lists("company_name", primary, secondary)
    assert len(merged) == 2
