import logging
from typing import List, Dict, Any, Optional
from fuzzywuzzy import fuzz
from src.extraction.regex_extractor import RegexExtractor
from src.extraction.spacy_extractor import SpacyExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridExtractor:
    """
    Combines results from RegexExtractor and SpacyExtractor with conflict resolution
    and confidence aggregation.
    """

    def __init__(self, spacy_model: str = "en_core_web_lg"):
        self.regex_extractor = RegexExtractor()
        try:
            self.spacy_extractor = SpacyExtractor(model_name=spacy_model)
        except Exception as e:
            logger.error(f"Failed to initialize SpacyExtractor: {e}")
            self.spacy_extractor = None

        # Define primary extractors for each field type
        self.field_priorities = {
            "email": "regex",
            "phone_number": "regex",
            "date": "regex",
            "amount": "regex",
            "invoice_number": "regex",
            "url": "regex",
            "person_name": "spacy",
            "company_name": "spacy",
            "address": "spacy",
            "job_title": "spacy",
            "skill": "spacy"
        }

    def extract_all(self, text: str) -> Dict[str, Any]:
        """
        Run all extractions and merge results.
        
        Args:
            text: Document text.
            
        Returns:
            Dictionary of consolidated extracted fields.
        """
        if not text:
            return {}

        merged_results = {}

        # 1. Run Regex Extraction
        regex_results = {
            "email": self.regex_extractor.extract_emails(text),
            "phone_number": self.regex_extractor.extract_phone_numbers(text),
            "date": self.regex_extractor.extract_dates(text),
            "amount": self.regex_extractor.extract_amounts(text),
            "invoice_number": self.regex_extractor.extract_invoice_number(text),
            "url": self.regex_extractor.extract_urls(text)
        }

        # 2. Run SpaCy Extraction
        if self.spacy_extractor:
            spacy_raw_entities = self.spacy_extractor.extract_entities(text)
            spacy_companies = self.spacy_extractor.extract_company_names(text)
            
            # Helper to flatten Spacy company dict
            all_companies = spacy_companies.get("vendor", []) + spacy_companies.get("customer", [])
            
            spacy_results = {
                "person_name": self.spacy_extractor.extract_person_names(text),
                "company_name": all_companies,
                "address": self.spacy_extractor.extract_addresses(text),
                "job_title": self.spacy_extractor.extract_job_titles(text),
                "skill": self.spacy_extractor.extract_skills(text)
            }
        else:
            spacy_results = {}

        # 3. Merge by Field Type
        all_keys = set(regex_results.keys()) | set(spacy_results.keys())
        
        for key in all_keys:
            r_list = regex_results.get(key, [])
            s_list = spacy_results.get(key, [])
            
            # Determine primary source logic
            primary_source = self.field_priorities.get(key, "regex") # Default to regex if unknown
            
            if primary_source == "regex":
                merged = self._merge_lists(job_name=key, primary=r_list, secondary=s_list)
            else:
                merged = self._merge_lists(job_name=key, primary=s_list, secondary=r_list)
                
            merged_results[key] = merged

        return merged_results

    def _merge_lists(self, job_name: str, primary: List[Dict], secondary: List[Dict]) -> List[Dict]:
        """
        Merge two lists of results with deduplication and confidence aggregation.
        
        Args:
            job_name: Name of the field being merged (for logging).
            primary: List of results from the prioritized extractor.
            secondary: List of results from the secondary extractor.
        """
        final_list = list(primary) # Start with all primary results
        
        for sec_item in secondary:
            is_duplicate = False
            best_match_idx = -1
            best_score = 0
            
            # check against all current final items
            for idx, pri_item in enumerate(final_list):
                # Fuzzy match values
                val1 = str(pri_item['value'])
                val2 = str(sec_item['value'])
                
                similarity = fuzz.ratio(val1.lower(), val2.lower())
                
                # Check for overlap in position if available
                pos_overlap = False
                if pri_item.get('position') and sec_item.get('position'):
                    p1 = pri_item['position']
                    p2 = sec_item['position']
                    # Simple overlap check: start1 <= end2 and start2 <= end1
                    if p1['start'] < p2['end'] and p2['start'] < p1['end']:
                        pos_overlap = True

                # Deduplication condition: High similarity OR position overlap
                if similarity > 85 or pos_overlap:
                    is_duplicate = True
                    if similarity > best_score:
                        best_score = similarity
                        best_match_idx = idx
            
            if is_duplicate and best_match_idx != -1:
                # Aggregate Confidence
                # Formula: 0.7 * primary + 0.3 * secondary
                # We assume 'pri_item' is primary because valid primary items are already in final_list 
                # (unless we appended a secondary item previously, but for simplicity:
                # logic dictates primary items preserve their "primary-ness").
                
                pri_item = final_list[best_match_idx]
                
                c_primary = pri_item['confidence']
                c_secondary = sec_item['confidence']
                
                new_conf = (0.7 * c_primary) + (0.3 * c_secondary)
                pri_item['confidence'] = round(min(new_conf, 1.0), 4) # Cap at 1.0
                
                # Conflict Resolution: Primary Value wins, so we don't overwrite 'value' 
                # unless secondary has significantly more info? 
                # Rule: "Conflict resolution when methods disagree" -> maintain primary value.
                
                # Optional: Merge metadata if needed
                
            else:
                # Unique item, add to list (but penalize confidence slightly as it wasn't corroborated?)
                # Strategy: Keep original confidence.
                final_list.append(sec_item)
                
        return final_list
