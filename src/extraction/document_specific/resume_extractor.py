import re
import logging
from typing import Dict, Any, List, Optional
from src.extraction.base import BaseExtractor
from src.extraction.regex_extractor import RegexExtractor
from src.extraction.spacy_extractor import SpacyExtractor

logger = logging.getLogger(__name__)

class ResumeExtractor(BaseExtractor):
    """
    Specialized extractor for Resumes/CVs.
    Identifies sections (Education, Experience, Skills) and extracts relevant entities.
    """

    def __init__(self, spacy_model: str = "en_core_web_lg"):
        super().__init__()
        self.regex_extractor = RegexExtractor()
        try:
            self.spacy_extractor = SpacyExtractor(model_name=spacy_model)
        except Exception as e:
            logger.error(f"Failed to initialize SpacyExtractor: {e}")
            self.spacy_extractor = None

    def extract(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Extract all resume fields from text.
        """
        if not text:
            logger.warning("Empty text provided to ResumeExtractor")
            return {}

        logger.info("Starting resume extraction")
        
        # 1. Split into sections
        sections = self._find_sections(text)
        
        # 2. Extract Fields
        contact = self.extract_contact_info(text) # Look everywhere for contact info
        education = self.extract_education(sections.get("Education", ""))
        experience = self.extract_work_experience(sections.get("Experience", ""))
        skills = self.extract_skills(sections.get("Skills", "") or text) # Fallback to full text if no section
        certifications = self.extract_certifications(sections.get("Certifications", ""))
        
        result = {
            "contact_info": contact,
            "education": education,
            "work_experience": experience,
            "skills": skills,
            "certifications": certifications,
            "document_type": "resume"
        }
        
        return result

    def _find_sections(self, text: str) -> Dict[str, str]:
        """
        Segment text into logical sections based on headers.
        """
        sections = {}
        
        # Common headers
        headers = {
            "Education": r"(?i)\b(Education|Academic Background|Qualifications)\b",
            "Experience": r"(?i)\b(Experience|Work Experience|Employment|History|Professional Experience)\b",
            "Skills": r"(?i)\b(Skills|Technical Skills|Competencies|Abilities|Technologies)\b",
            "Certifications": r"(?i)\b(Certifications|Certificates|Awards|Honors)\b",
            "Projects": r"(?i)\b(Projects|Portfolio)\b"
        }
        
        # Find indices of all headers
        found_indices = []
        for section, pattern in headers.items():
            for match in re.finditer(pattern, text):
                # Heuristic: Header should be on its own line or start of line
                # and usually short (less than 5 words)
                line_start = text.rfind('\n', 0, match.start())
                if line_start == -1: line_start = 0
                line_end = text.find('\n', match.end())
                if line_end == -1: line_end = len(text)
                
                line_content = text[line_start:line_end].strip()
                
                # Check if plausible header (short, mostly proper case or UPPER)
                if len(line_content.split()) < 6:
                    found_indices.append((match.start(), section))
        
        # Sort by position
        found_indices.sort()
        
        # Extract content between headers
        for i, (start, section_name) in enumerate(found_indices):
            content_start = start
            if i + 1 < len(found_indices):
                content_end = found_indices[i+1][0]
            else:
                content_end = len(text)
            
            # Combine if section already exists (e.g. "Experience" appears twice?) - usually overwrite or append
            # Let's append if exists to be safe
            current_content = text[content_start:content_end].strip()
            if section_name in sections:
                sections[section_name] += "\n" + current_content
            else:
                sections[section_name] = current_content
                
        return sections

    def extract_contact_info(self, text: str) -> Dict[str, Any]:
        """
        Extract contact details (Name, Email, Phone, Links)
        """
        contact = {}
        
        # Name
        # Heuristic: Name is usually at the very top
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 2 and len(line.split()) < 5:
                # Basic name validity check (no digits, etc)
                if not any(char.isdigit() for char in line):
                     if self.spacy_extractor:
                         # Use Spacy to confirm PERSON
                         names = self.spacy_extractor.extract_person_names(line)
                         if names:
                             contact["name"] = names[0]
                             break
                     else:
                         # Blind guess
                         contact["name"] = self._create_field(line, "string", 0.5, "heuristic_top")
                         break
                         
        # Email
        emails = self.regex_extractor.extract_emails(text)
        if emails:
            contact["email"] = emails[0]
            
        # Phone
        phones = self.regex_extractor.extract_phone_numbers(text)
        if phones:
            contact["phone"] = phones[0]
            
        # Links (LinkedIn, GitHub, Portfolio)
        urls = self.regex_extractor.extract_urls(text)
        links = []
        for url_field in urls:
            url = url_field["value"].lower()
            if "linkedin" in url:
                contact["linkedin"] = url_field
            elif "github" in url:
                contact["github"] = url_field
            else:
                links.append(url_field)
        if links:
            contact["other_links"] = links
            
        return contact

    def extract_education(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract education details.
        """
        edu_entries = []
        if not text:
            return []
            
        # Strategy: Look for degree keywords, then find Org (University) and Dates nearby
        # Expanded pattern to capture "Bachelor of Science", "Master in X", etc.
        # Flatter structure: Degree + (space + keyword)*
        degree_patterns = r"(?i)\b(?:B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|Ph\.?D\.?|Bachelor|Master|Doctor|Diploma|Certificate|Associate)(?:\s+(?:of|in|to|Science|Arts|Engineering|Business|Technology|Management|Education|Philosophy|Computer))+\b"
        
        # We might want to iterate line by line or chunks?
        # Let's just create one entry per "Degree" found
        
        for match in re.finditer(degree_patterns, text):
            # For each degree, define a window to find School and Date
            start, end = match.span()
            # Look ahead and behind a bit, or just take the line
            line_start = text.rfind('\n', 0, start)
            line_end = text.find('\n', end)
            line_content = text[max(0, line_start):line_end].strip() # The text containing degree
            
            entry = {
                "degree": self._create_field(match.group(), "string", 0.8, "regex_keyword"),
                "raw_text": line_content
            }
            
            # Find University (ORG)
            if self.spacy_extractor:
                # Expand context for school search (next/prev line)
                context_start = max(0, text.rfind('\n', 0, max(0, line_start-1)))
                context_end = text.find('\n', line_end+1) 
                if context_end == -1: context_end = len(text)
                context_text = text[context_start:context_end]
                
                ents = self.spacy_extractor.extract_entities(context_text).get("ORG", [])
                if ents:
                    # Pick the one that looks most like a university
                    for ent in ents:
                        name = ent["value"].lower()
                        if any(k in name for k in ["univ", "college", "school", "institute"]):
                            entry["institution"] = ent
                            break
                    if "institution" not in entry and ents:
                        # Fallback to first ORG
                        entry["institution"] = ents[0]

            # Find Year (####)
            dates = re.findall(r"\b(19|20)\d{2}\b", line_content) # Simple year extraction
            if not dates:
                 # Check next line too
                 next_line_end = text.find('\n', line_end+1)
                 next_line = text[line_end:next_line_end] if next_line_end != -1 else ""
                 dates = re.findall(r"\b(19|20)\d{2}\b", next_line)
                 
            if dates:
                # If multiple dates (start-end), typically the last one is grad year
                entry["year"] = self._create_field(dates[-1], "year", 0.8, "regex")

            edu_entries.append(entry)
            
        return edu_entries

    def extract_work_experience(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract work experience.
        """
        experiences = []
        if not text:
            return []
            
        # Deep extraction of experience is hard without visual structure.
        # We can try to rely on Spacy to find ORGs (Companies) and Job Titles.
        
        # Use Spacy structure
        if self.spacy_extractor:
            titles = self.spacy_extractor.extract_job_titles(text)
            # Group by proximity to ORGs?
            
            # Simplified approach: Iterate through titles found, assume they head a block
            for i, title in enumerate(titles):
                entry = {
                    "title": title
                }
                
                # Context window around title
                start_idx = title.get("position", {}).get("start", 0)
                
                # Look for Company (ORG) in same line or previous line
                line_start = text.rfind('\n', 0, start_idx)
                if line_start == -1: line_start = 0
                
                prev_line_start = text.rfind('\n', 0, max(0, line_start-1))
                context = text[max(0, prev_line_start):text.find('\n', start_idx+100)] # generous window
                
                ents = self.spacy_extractor.extract_entities(context).get("ORG", [])
                if ents:
                    entry["company"] = ents[0] # Assume first ORG nearby is company
                    
                # Look for Dates
                date_matches = self.regex_extractor.extract_dates(context)
                if date_matches:
                    entry["dates"] = date_matches # Store all dates found in context
                    
                experiences.append(entry)
        
        return experiences

    def extract_skills(self, text: str) -> Dict[str, List[Any]]:
        """
        Extract and categorize skills.
        """
        skills_data = {"all": []}
        if not text:
            return skills_data
            
        if self.spacy_extractor:
            found_skills = self.spacy_extractor.extract_skills(text)
            skills_data["all"] = found_skills
            
            # Simple Categorization (could be improved with a knowledge base)
            categories = {
                "Languages": ["Python", "Java", "C++", "JavaScript", "TypeScript", "Go", "Rust", "SQL"],
                "Web": ["React", "Angular", "Vue", "Node", "HTML", "CSS", "Django", "Flask", "FastAPI"],
                "Cloud/DevOps": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Jenkins"],
                "Data/AI": ["Machine Learning", "Deep Learning", "Pandas", "NumPy", "TensorFlow", "PyTorch"]
            }
            
            skills_data["categorized"] = {}
            for cat, keywords in categories.items():
                skills_data["categorized"][cat] = []
                for skill in found_skills:
                    # simplistic substring match or exact match
                    val = skill["value"]
                    if any(k.lower() == val.lower() or (len(k)>2 and k.lower() in val.lower()) for k in keywords):
                        skills_data["categorized"][cat].append(skill)
                        
        return skills_data

    def extract_certifications(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract certifications.
        """
        certs = []
        if not text:
            return []
            
        # Line-based heuristic: assume each line in this section is a cert
        lines = text.split('\n')
        # Skip header if included in text (though _find_sections includes distinct content usually)
        
        for line in lines:
            line = line.strip()
            if len(line) < 5 or "certification" in line.lower():
                continue
                
            certs.append({
                "name": self._create_field(line, "string", 0.6, "line_item")
            })
            
        return certs
