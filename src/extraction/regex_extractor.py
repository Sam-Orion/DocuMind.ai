import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import dateutil.parser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegexExtractor:
    """
    Extracts structured data (emails, phones, dates, amounts, etc.) using regular expressions.
    """

    def __init__(self):
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            # US and International phone format approximation
            "phone": r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:ext|x)\s*\d+)?',
            # Capture various date formats: MM/DD/YYYY, YYYY-MM-DD, DD-Mon-YYYY
            "date": r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-.\s,]\d{1,2}[-.\s,]\d{4})\b',
            # Amounts: Currency symbols optionally, followed by numbers with commas/decimals
            "amount": r'(?:[\$₹€£]|USD|INR|EUR)?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b',
            # Invoice Numbers: Look for keywords like "Invoice #" followed by alphanumeric
            "invoice_number": r'(?:Invoice|Bill|Reference|Inv)\s*(?:No|Number|#)?\s*[:.-]?\s*([A-Za-z0-9/-]+)',
            "url": r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        }

    def _format_result(self, value: Any, confidence: float, field_type: str, original_text: str, start: int = -1, end: int = -1) -> Dict[str, Any]:
        """Helper to standardize return format."""
        return {
            "value": value,
            "confidence": confidence,
            "field_type": field_type,
            "position": {"start": start, "end": end} if start != -1 else None,
            "original_text": original_text
        }

    def extract_emails(self, text: str) -> List[Dict[str, Any]]:
        """Extract email addresses."""
        results = []
        for match in re.finditer(self.patterns["email"], text):
            results.append(self._format_result(
                value=match.group(),
                confidence=1.0, # Regex matches are usually definite if highly specific
                field_type="email",
                original_text=match.group(),
                start=match.start(),
                end=match.end()
            ))
        return results

    def extract_phone_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract phone numbers."""
        results = []
        for match in re.finditer(self.patterns["phone"], text):
            # Basic validation to avoid false positives (e.g. date parts)
            phone = match.group()
            if sum(c.isdigit() for c in phone) >= 7:
                results.append(self._format_result(
                    value=phone.strip(),
                    confidence=0.85, 
                    field_type="phone_number",
                    original_text=phone,
                    start=match.start(),
                    end=match.end()
                ))
        return results

    def extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract and normalize dates to ISO format (YYYY-MM-DD)."""
        results = []
        for match in re.finditer(self.patterns["date"], text, re.IGNORECASE):
            date_str = match.group()
            try:
                # Normalize using dateutil
                dt = dateutil.parser.parse(date_str, fuzzy=False)
                iso_date = dt.strftime("%Y-%m-%d")
                results.append(self._format_result(
                    value=iso_date,
                    confidence=0.90,
                    field_type="date",
                    original_text=date_str,
                    start=match.start(),
                    end=match.end()
                ))
            except (ValueError, OverflowError):
                logger.warning(f"Extracted date string '{date_str}' could not be parsed.")
                continue
        return results

    def extract_amounts(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary amounts and values."""
        results = []
        # Use a more specific regex loop to capture groups correctly
        # Regex captures: Group 1 is the number part
        for match in re.finditer(self.patterns["amount"], text):
            amount_str = match.group(1) 
            full_match = match.group(0)
            
            if not amount_str: 
                continue

            try:
                # Remove commas for float conversion
                clean_value = float(amount_str.replace(',', ''))
                # Filter out likely non-amounts (e.g. years)
                if clean_value > 1900 and clean_value < 2100 and "." not in amount_str:
                     # Could be a year, lower confidence or skip?
                     confidence = 0.5
                else:
                    confidence = 0.9

                if "." not in amount_str:
                     # Integers might look like other IDs
                     confidence -= 0.1

                results.append(self._format_result(
                    value=clean_value,
                    confidence=confidence,
                    field_type="amount",
                    original_text=full_match,
                    start=match.start(),
                    end=match.end()
                ))
            except ValueError:
                continue
        return results

    def extract_invoice_number(self, text: str) -> List[Dict[str, Any]]:
        """Extract invoice numbers based on context."""
        results = []
        for match in re.finditer(self.patterns["invoice_number"], text, re.IGNORECASE):
            inv_num = match.group(1).strip()
            # Invoice numbers usually have digits
            if re.search(r'\d', inv_num):
                 results.append(self._format_result(
                    value=inv_num,
                    confidence=0.85,
                    field_type="invoice_number",
                    original_text=match.group(0),
                    start=match.start(),
                    end=match.end()
                ))
        return results

    def extract_urls(self, text: str) -> List[Dict[str, Any]]:
        """Extract URLs."""
        results = []
        for match in re.finditer(self.patterns["url"], text, re.IGNORECASE):
            results.append(self._format_result(
                value=match.group(),
                confidence=0.95,
                field_type="url",
                original_text=match.group(),
                start=match.start(),
                end=match.end()
            ))
        return results
