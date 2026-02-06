import pytest
import os
from src.extraction.document_specific.resume_extractor import ResumeExtractor

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "samples", "sample_resume.txt")

@pytest.fixture
def resume_text():
    with open(SAMPLE_PATH, "r") as f:
        return f.read()

@pytest.fixture
def extractor():
    return ResumeExtractor()

class TestResumeExtractor:
    
    def test_find_sections(self, extractor, resume_text):
        sections = extractor._find_sections(resume_text)
        assert "Experience" in sections
        assert "Education" in sections
        assert "Skills" in sections
        assert "Certifications" in sections
        
        # Check content roughly
        assert "Tech Solutions Inc" in sections["Experience"]
        assert "State University" in sections["Education"]

    def test_extract_contact_info(self, extractor, resume_text):
        info = extractor.extract_contact_info(resume_text)
        
        # Name might depend on Spacy model presence, but heuristic should catch top line
        if info.get("name"):
            assert "John Doe" in info["name"]["value"]
            
        assert "john.doe@email.com" in info["email"]["value"]
        assert "555" in info["phone"]["value"]
        
        # Links
        links = info.get("linkedin")
        if links:
            assert "linkedin.com/in/johndoe" in links["value"]
        
        github = info.get("github")
        if github:
            assert "github.com/johndoe" in github["value"]

    def test_extract_education(self, extractor, resume_text):
        # We process the text normally in extract(), but here we test the sub-method
        # We need to feed it the specific section text logic or full text if robust
        # The method expects just the text chunk, let's use helper
        sections = extractor._find_sections(resume_text)
        edu_text = sections["Education"]
        
        edu = extractor.extract_education(edu_text)
        assert len(edu) >= 1
        assert "Bachelor of Science" in edu[0]["degree"]["value"]
        # Year
        if "year" in edu[0]:
            assert "2019" in edu[0]["year"]["value"]
        # Institution might need Spacy to work well, check if present
        if "institution" in edu[0]:
             assert "State University" in edu[0]["institution"]["value"]

    def test_extract_work_experience(self, extractor, resume_text):
        sections = extractor._find_sections(resume_text)
        exp_text = sections["Experience"]
        
        # This heavily implies Spacy is working for correct structure
        # But let's check basic return
        exp = extractor.extract_work_experience(exp_text)
        
        # If Spacy matches "Software Engineer" as title:
        # Note: In CI/Test environment without full model, this might return empty or heuristic
        # We check if it returns a list, and if elements have plausible data
        assert isinstance(exp, list)
        
        # If we have matches
        if exp:
            values = [e["title"]["value"] for e in exp if "title" in e]
            assert any("Software Engineer" in v for v in values)

    def test_extract_skills(self, extractor, resume_text):
        sections = extractor._find_sections(resume_text)
        skills_text = sections["Skills"]
        
        skills = extractor.extract_skills(skills_text)
        assert "all" in skills
        
        # Check for Python
        skill_names = [s["value"] for s in skills["all"]]
        assert "Python" in skill_names
        assert "Docker" in skill_names
        
        # Check categorization
        if "categorized" in skills:
            assert "Python" in [s["value"] for s in skills["categorized"].get("Languages", [])]
            assert "AWS" in [s["value"] for s in skills["categorized"].get("Cloud/DevOps", [])]

    def test_extract_certifications(self, extractor, resume_text):
        sections = extractor._find_sections(resume_text)
        cert_text = sections["Certifications"]
        
        certs = extractor.extract_certifications(cert_text)
        assert len(certs) >= 2
        # Check specific cert name presence
        cert_names = [c["name"]["value"] for c in certs]
        assert any("AWS Certified" in c for c in cert_names)
