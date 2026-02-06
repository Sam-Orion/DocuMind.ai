import re
import logging
from typing import Optional, List, Any
import dateutil.parser
import phonenumbers

logger = logging.getLogger(__name__)

class AutoCorrector:
    """
    Normalizes extracted data and correct common OCR errors.
    """

    @staticmethod
    def correct_date_format(date_str: str) -> Optional[str]:
        """
        Normalize various date formats to ISO 8601 (YYYY-MM-DD).
        """
        if not date_str:
            return None
        try:
            # Fuzzy parsing allows strings like "Due: Jan 5, 2023" to be parsed
            dt = dateutil.parser.parse(date_str, fuzzy=True)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            logger.debug(f"Could not parse date: {date_str}")
            return None

    @staticmethod
    def correct_amount_format(amount_str: str) -> Optional[float]:
        """
        Normalize currency amounts.
        """
        if not amount_str:
            return None
            
        # 1. Clean up known non-numeric junk
        # Remove currency symbols and valid text
        # Keep digits, dots, commas, minus sign
        cleaned = re.sub(r"[^\d.,-]", "", str(amount_str))
        
        # 2. Handle common OCR char swaps for numbers if it looks like a number context
        # 'O' -> '0', 'l' -> '1', 'S' -> '5' (dangerous if mixed with text, but amount_str assumes prior field extraction)
        # But for strictly numeric fields, it's often safer to rely on regex extractor. 
        # Here we assume extraction happened, so we just parse standard formats.
        
        try:
            # If comma is decimal separator (10,00) vs thousands separator (10,000.00) makes it tricky
            # Heuristic: if '.' in string, assume dot is decimal unless multiple dots.
            # If ',' in string and '.' not in string, look at position.
            
            # Simple US/UK Centric: Remove ',' entirely, keep '.'
            # (TODO: Add locale support for EU style 1.000,00)
            
            # Robust check for comma usage
            if ',' in cleaned and '.' in cleaned:
                # 1,234.56 -> 1234.56
                if cleaned.rfind(',') < cleaned.rfind('.'):
                    cleaned = cleaned.replace(',', '')
                else: 
                    # 1.234,56 -> 1234.56
                    cleaned = cleaned.replace('.', '').replace(',', '.')
            elif ',' in cleaned:
                # 100,50 -> 100.50 (if looking like decimal) or 100,000 -> 100000
                if len(cleaned.split(',')[-1]) == 2:
                     cleaned = cleaned.replace(',', '.')
                else:
                     cleaned = cleaned.replace(',', '')
                     
            return float(cleaned)
        except ValueError:
            return None

    @staticmethod
    def correct_phone_format(phone_str: str, region: str = "US") -> Optional[str]:
        """
        Normalize phone number to E.164.
        """
        if not phone_str:
            return None
        try:
            parsed = phonenumbers.parse(phone_str, region)
            if phonenumbers.is_valid_number(parsed):
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
            return None
        except phonenumbers.NumberParseException:
            return None

    @staticmethod
    def suggest_corrections(value: str, field_type: str, confidence: float) -> List[Any]:
        """
        Suggest list of possible correct values if confidence is low.
        Current version uses simple heuristics.
        """
        suggestions = []
        
        if field_type == "amount" or field_type == "currency":
            # Check for common OCR confusions: S -> 5, O -> 0, B -> 8
            # Only apply if value is mixed alphanumeric but expected numeric
            replacements = {'O': '0', 'o': '0', 'S': '5', 's': '5', 'B': '8', 'l': '1', 'I': '1', 'Z': '2'}
            new_val = value
            for char, rep in replacements.items():
                new_val = new_val.replace(char, rep)
                
            corrected = AutoCorrector.correct_amount_format(new_val)
            if corrected is not None and str(corrected) != value:
                 suggestions.append(corrected)
                 
        if field_type == "date":
            # Attempt to parse
            corrected = AutoCorrector.correct_date_format(value)
            if corrected and corrected != value:
                suggestions.append(corrected)
                
        return suggestions
