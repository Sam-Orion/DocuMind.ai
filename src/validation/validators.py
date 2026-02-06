import re
import datetime
import phonenumbers
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class FieldValidator:
    """
    Validates individual field values.
    """
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format and domain.
        """
        if not email:
            return False
        # Basic regex
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(pattern, email):
            return False
        
        # Domain check (blacklist example)
        domain = email.split('@')[-1]
        if domain in ["example.com", "test.com"]:
            return False
            
        return True

    @staticmethod
    def validate_phone_number(phone: str, region: str = "US") -> bool:
        """
        Validate phone number using phonenumbers library.
        """
        if not phone:
            return False
        try:
            parsed = phonenumbers.parse(phone, region)
            return phonenumbers.is_valid_number(parsed)
        except phonenumbers.NumberParseException:
            return False

    @staticmethod
    def validate_date(date_str: str) -> bool:
        """
        Validate date is parsable and within reasonable range (1900 - Future+10y).
        Assumes ISO format YYYY-MM-DD or simple formats if parsed differently.
        Here we assume the extracted value is already normalized to YYYY-MM-DD or is a string we check.
        """
        if not date_str:
            return False
        
        # Check ISO format
        try:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return False
            
        year = dt.year
        current_year = datetime.datetime.now().year
        
        if 1900 <= year <= current_year + 10:
            return True
        return False

    @staticmethod
    def validate_amount(amount: Any) -> bool:
        """
        Validate monetary amount (non-negative, not excessively large).
        """
        if amount is None:
            return False
        try:
            val = float(amount)
            if val < 0:
                return False
            if val > 1_000_000_000: # 1 Billion cap for sanity
                logger.warning(f"Amount {val} seems excessively large.")
                return True # Technically valid but suspicious
            return True
        except ValueError:
            return False

class InvoiceValidator:
    """
    Validates invoice logic.
    """
    
    @staticmethod
    def validate_totals(data: Dict[str, Any], tolerance: float = 0.05) -> Dict[str, Any]:
        """
        Check if Subtotal + Tax + Shipping - Discount == Total.
        Returns validation result dict.
        """
        validation_results = {"valid": True, "errors": []}
        
        def get_val(key):
            field = data.get(key)
            if field and isinstance(field, dict):
                # Handle simplified field extraction where value might be string with currency symbols
                val = field.get("value", 0.0)
                if isinstance(val, str):
                    val = val.replace("$", "").replace(",", "")
                try:
                    return float(val)
                except ValueError:
                    return 0.0
            return 0.0
            
        subtotal = get_val("subtotal")
        tax = get_val("tax")
        shipping = get_val("shipping")
        discount = get_val("discount")
        total_amount = get_val("total_amount")
        
        # Only validate if we have at least subtotal and total
        if subtotal > 0 and total_amount > 0:
            calculated_total = subtotal + tax + shipping - discount
            
            if abs(calculated_total - total_amount) > tolerance:
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"Total mismatch: Calculated {calculated_total:.2f} != Extracted {total_amount:.2f}"
                )
            
        return validation_results

    @staticmethod
    def validate_dates(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check Due Date >= Invoice Date.
        """
        validation_results = {"valid": True, "errors": []}
        
        def get_date(key):
            field = data.get(key)
            if field and isinstance(field, dict):
                val = field.get("value")
                if FieldValidator.validate_date(val):
                    return datetime.datetime.strptime(val, "%Y-%m-%d")
            return None
            
        inv_date = get_date("invoice_date")
        due_date = get_date("due_date")
        
        if inv_date and due_date:
            if due_date < inv_date:
                validation_results["valid"] = False
                validation_results["errors"].append(
                    f"Due Date {due_date.date()} is before Invoice Date {inv_date.date()}"
                )
                
        return validation_results

class ResumeValidator:
    """
    Validates resume logic.
    """
    @staticmethod
    def validate_dates(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for logical date consistencies.
        """
        # Placeholder
        return {"valid": True, "errors": []}

class CrossFieldValidator:
    """
    Runs all appropriate validators for a document type.
    """
    
    @staticmethod
    def validate(data: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """
        Main validation entry point.
        """
        report = {
            "document_valid": True,
            "field_validations": {},
            "logic_validations": []
        }
        
        # 1. Field Validations (Generic)
        # Walk through common keys
        
        # Invoice/Receipt amounts
        for key in ["subtotal", "tax", "total_amount", "discount"]:
            if key in data and isinstance(data[key], dict):
                val = data[key].get("value")
                # Clean currency string
                if isinstance(val, str):
                    val = val.replace("$", "").replace(",", "")
                is_valid = FieldValidator.validate_amount(val)
                report["field_validations"][key] = is_valid
                if not is_valid: report["document_valid"] = False

        # Contact info check
        if "contact_info" in data and isinstance(data["contact_info"], dict):
            ci = data["contact_info"]
            if "email" in ci and isinstance(ci["email"], dict):
                is_valid = FieldValidator.validate_email(ci["email"].get("value"))
                report["field_validations"]["email"] = is_valid
            if "phone" in ci and isinstance(ci["phone"], dict):
                is_valid = FieldValidator.validate_phone_number(ci["phone"].get("value"))
                report["field_validations"]["phone"] = is_valid
                
        # 2. Logic Validations
        if document_type == "invoice":
            totals = InvoiceValidator.validate_totals(data)
            if not totals["valid"]:
                report["document_valid"] = False
                report["logic_validations"].extend(totals["errors"])
                
            dates = InvoiceValidator.validate_dates(data)
            if not dates["valid"]:
                report["document_valid"] = False
                report["logic_validations"].extend(dates["errors"])
                
        return report
