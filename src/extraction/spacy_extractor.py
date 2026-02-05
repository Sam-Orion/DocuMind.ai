import spacy
import logging
import re
from typing import List, Dict, Any, Optional, Set
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpacyExtractor:
    """
    Extracts structured data (entities, names, addresses, skills) using spaCy NER and pattern matching.
    """

    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize the SpacyExtractor with a specific model.
        
        Args:
            model_name: The name of the spaCy model to load. Defaults to "en_core_web_lg".
        """
        try:
            logger.info(f"Loading spaCy model: {model_name}")
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Model '{model_name}' not found. Downloading...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)
        except Exception as e:
            logger.error(f"Failed to load spaCy model '{model_name}': {e}")
            raise

        # Initialize matchers
        self.matcher = PhraseMatcher(self.nlp.vocab)
        self._initialize_skills_matcher()

    def _initialize_skills_matcher(self):
        """Initialize the PhraseMatcher with a predefined list of skills."""
        # MVP Skills list - in a real app, this would come from a database or file
        common_skills = [
            "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
            "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "Redis",
            "React", "Angular", "Vue", "Next.js", "Node.js", "Django", "FastAPI", "Flask",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch", "Scikit-learn",
            "Project Management", "Agile", "Scrum", "Communication", "Leadership"
        ]
        patterns = [self.nlp.make_doc(text) for text in common_skills]
        self.matcher.add("SKILL", patterns)

    def _format_result(self, value: Any, confidence: float, field_type: str, start: int = -1, end: int = -1, label: str = "") -> Dict[str, Any]:
        """Helper to standardize return format."""
        if isinstance(value, str):
            value = value.strip()
            
        return {
            "value": value,
            "confidence": confidence,
            "field_type": field_type,
            "entity_label": label, # e.g. PERSON, ORG
            "position": {"start": start, "end": end} if start != -1 else None
        }

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract standard named entities (PERSON, ORG, GPE, DATE, MONEY).
        
        Args:
            text: The text to process.
            
        Returns:
            Dictionary of extracted entities grouped by type.
        """
        if not text:
            return {}

        doc = self.nlp(text)
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": [],
            "MONEY": []
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(self._format_result(
                    value=ent.text,
                    confidence=1.0, # spaCy generally produces high confidence for recognized ents
                    field_type="entity",
                    start=ent.start_char,
                    end=ent.end_char,
                    label=ent.label_
                ))
        
        return entities

    def extract_person_names(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract and filter person names to reduce false positives.
        
        Args:
            text: The text to process.
            
        Returns:
            List of valid person names.
        """
        if not text:
            return []

        doc = self.nlp(text)
        persons = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Filter: Names should usually have at least two parts (First Last) 
                # or should not be common stop words.
                if len(ent.text.split()) > 1 or (len(ent.text) > 3 and ent.text[0].isupper()):
                     persons.append(self._format_result(
                        value=ent.text,
                        confidence=0.9,
                        field_type="person_name",
                        start=ent.start_char,
                        end=ent.end_char,
                        label="PERSON"
                    ))
        
        return persons

    def extract_company_names(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract company names and attempt to classify them (Vendor vs Customer).
        
        Args:
            text: The text to process.
            
        Returns:
            Dictionary with 'vendor' and 'customer' keys.
        """
        if not text:
            return {"vendor": [], "customer": []}

        doc = self.nlp(text)
        companies = {"vendor": [], "customer": []}
        
        # Heuristic: First ORG is often the Vendor/Sender in invoices
        # Heuristic: ORG near "Bill To" or "Ship To" is Customer
        
        first_org_found = False
        lines = text.split('\n')
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                # Context check
                start_idx = ent.start_char
                # Look at window before entity
                pre_window = text[max(0, start_idx-50):start_idx].lower()
                
                is_customer = False
                if "bill to" in pre_window or "ship to" in pre_window or "customer" in pre_window:
                    is_customer = True
                    confidence = 0.85
                elif not first_org_found:
                    # Assume first ORG is vendor if no specific context
                     is_customer = False
                     confidence = 0.7
                     first_org_found = True
                else:
                    # Default bucket if unknown
                    is_customer = False # Treating others as potential vendors/partners
                    confidence = 0.5

                category = "customer" if is_customer else "vendor"
                
                companies[category].append(self._format_result(
                    value=ent.text,
                    confidence=confidence,
                    field_type="company_name",
                    start=ent.start_char,
                    end=ent.end_char,
                    label="ORG"
                ))

        return companies

    def extract_addresses(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract addresses using GPE entities and regex patterns for ZIP usage.
        
        Args:
            text: The text to process.
            
        Returns:
            List of extracted addresses.
        """
        if not text:
            return []

        doc = self.nlp(text)
        addresses = []
        
        # Regex for US ZIP codes (5 digits, optional -4)
        zip_pattern = r'\b\d{5}(?:-\d{4})?\b'
        
        # Strategy: Look for GPE (Geopolitical Entity) and see if it looks like part of an address
        # Or look for ZIP codes and expand window
        
        # 1. Capture lines with GPEs that also have numbers (street num or zip)
        for ent in doc.ents:
            if ent.label_ == "GPE":
                # Check immediate context
                # This is a simplification; full address parsing is complex.
                # We'll look for lines containing the GPE.
                pass 

        # 2. Regex Search for ZIP codes as anchors
        for match in re.finditer(zip_pattern, text):
            zip_code = match.group()
            start, end = match.span()
            
            # Look backwards for City, State (often capture entire line)
            line_start = text.rfind('\n', 0, start)
            if line_start == -1: line_start = 0
            line_end = text.find('\n', end)
            if line_end == -1: line_end = len(text)
            
            full_line = text[line_start:line_end].strip()
            
            # Simple validation: line should contain letters
            if any(c.isalpha() for c in full_line):
                addresses.append(self._format_result(
                    value=full_line,
                    confidence=0.8,
                    field_type="address",
                    start=line_start,
                    end=line_end
                ))

        return addresses

    def extract_skills(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract skills from text (useful for resumes).
        
        Args:
            text: The text to process.
            
        Returns:
            List of extracted skills.
        """
        if not text:
            return []

        doc = self.nlp(text)
        matches = self.matcher(doc)
        skills = []
        
        for match_id, start, end in matches:
            span = doc[start:end]
            skills.append(self._format_result(
                value=span.text,
                confidence=1.0,
                field_type="skill",
                start=span.start_char,
                end=span.end_char
            ))
            
        # Deduplicate by value
        unique_skills = []
        seen = set()
        for skill in skills:
            if skill['value'] not in seen:
                seen.add(skill['value'])
                unique_skills.append(skill)
                
        return unique_skills

    def extract_job_titles(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract job titles using patterns.
        
        Args:
            text: The text to process.
            
        Returns:
            List of extracted job titles.
        """
        if not text:
            return []

        # Patterns: 
        # "Title at Company"
        # "Title - Company"
        # "Title, Company"
        
        # We can use spaCy dependency parsing or simple regex + NER
        # For MVP, let's use a regex with some common title keywords
        
        common_titles = r"(Software Engineer|Developer|Manager|Director|CTO|CEO|COO|Designer|Architect|Consultant|Analyst|Scientist|Coordinator|Administrator)"
        pattern = fr"\b({common_titles}(?:\s+[A-Za-z]+){{0,3}})\s+(?:at|@|for|-|with)\s+([A-Z][A-Za-z0-9\s]+)"
        
        titles = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            full_match = match.group(0)
            title = match.group(1)
            # Company part is match.group(2)
            
            titles.append(self._format_result(
                value=title,
                confidence=0.75,
                field_type="job_title",
                start=match.start(),
                end=match.end()
            ))
            
        return titles
