import re
import logging
from typing import Dict, Any, List, Optional
from src.extraction.base import BaseExtractor
from src.extraction.regex_extractor import RegexExtractor
from src.extraction.spacy_extractor import SpacyExtractor

logger = logging.getLogger(__name__)

class InvoiceExtractor(BaseExtractor):
    """
    Specialized extractor for Invoices.
    Extracts headers, line items, totals, and party information.
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
        Extract all invoice fields from text.
        """
        if not text:
            logger.warning("Empty text provided to InvoiceExtractor")
            return {}

        logger.info("Starting invoice extraction")
        
        # 1. Header Information
        header = self.extract_invoice_header(text)
        
        # 2. Parties (Vendor/Customer)
        parties = self.extract_parties(text)
        
        # 3. Line Items
        line_items = self.extract_line_items(text)
        
        # 4. Totals
        totals = self.extract_totals(text, line_items)
        
        # 5. Payment Terms
        payment_terms = self.extract_payment_terms(text)
        
        # Combine all results
        result = {
            "header": header,
            "vendor": parties.get("vendor", {}),
            "customer": parties.get("customer", {}),
            "line_items": line_items,
            "totals": totals,
            "payment_terms": payment_terms,
            "document_type": "invoice",  # Explicitly allow identification
            "validation_errors": totals.get("validation_errors", [])
        }
        
        return result

    def extract_invoice_header(self, text: str) -> Dict[str, Any]:
        """
        Extract invoice number, date, due date, PO number.
        """
        header_data = {}
        
        # Invoice Number
        inv_nums = self.regex_extractor.extract_invoice_number(text)
        if inv_nums:
            header_data["invoice_number"] = inv_nums[0]
        
        # Dates - simplistic approach: first date is likely invoice date, later is due date
        dates = self.regex_extractor.extract_dates(text)
        if dates:
            header_data["invoice_date"] = dates[0]
            if len(dates) > 1:
                # Need smarter logic, but for MVP, assume subsequent dates might be due dates 
                # especially if near "Due Date" keywords, but let's just grab the second one for now
                # or look for specific keywords
                header_data["due_date"] = dates[1]
                
        # PO Number regex
        # Use word boundaries to avoid matching inside words like "support"
        po_pattern = r"\b(?:P\.O\.|Purchase Order|PO Number|PO)\b\s*[:#]?\s*([A-Z0-9-]+)"
        po_match = re.search(po_pattern, text, re.IGNORECASE)
        if po_match:
            header_data["po_number"] = self._create_field(
                value=po_match.group(1),
                field_type="po_number",
                confidence=0.9,
                source="regex"
            )
            
        return header_data

    def extract_parties(self, text: str) -> Dict[str, Any]:
        """
        Extract vendor and customer information.
        """
        parties = {"vendor": {}, "customer": {}}
        
        if not self.spacy_extractor:
            return parties
            
        # Leverage Spacy for ORG and PERSON
        companies = self.spacy_extractor.extract_company_names(text)
        
        # Simplistic heuristic: First found company is often Vendor, second might be Customer
        # Or look for "Bill To" / "Ship To" sections
        
        # For MVP, let's trust the SpacyExtractor's simple heuristics or just take the list
        # But SpacyExtractor.extract_company_names returns {'vendor': [], 'customer': []} usually?
        # Let's check spacy_extractor.py ... actually I don't recall it returning that structure.
        # Let's assume it returns a list or a list of dicts. 
        # In hybrid_extractor it was used as: spacy_companies.get("vendor", []) which implies it might separate them.
        # But checking recent view_file of hybrid_extractor, it assumes extract_company_names returns a dict.
        # Let's stick to using what we extract generally.
        
        if isinstance(companies, dict):
             if companies.get("vendor"):
                 parties["vendor"]["name"] = companies["vendor"][0]
             if companies.get("customer"):
                 parties["customer"]["name"] = companies["customer"][0]
        elif isinstance(companies, list) and companies:
             # Fallback if list
             parties["vendor"]["name"] = companies[0]
             if len(companies) > 1:
                 parties["customer"]["name"] = companies[1]

        # Address extraction
        addresses = self.spacy_extractor.extract_addresses(text)
        if addresses:
            parties["vendor"]["address"] = addresses[0]
            if len(addresses) > 1:
                parties["customer"]["address"] = addresses[1]

        # Emails/Phones via Regex
        emails = self.regex_extractor.extract_emails(text)
        phones = self.regex_extractor.extract_phone_numbers(text)
        
        if emails:
            parties["vendor"]["email"] = emails[0]
        if phones:
             parties["vendor"]["phone"] = phones[0]

        return parties

    def extract_line_items(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract line items from Table structures.
        Currently uses a line-by-line regex heuristic.
        """
        items = []
        lines = text.split('\n')
        
        # Common header keywords
        header_pattern = r"(description|item|qty|quantity|rate|unit price|amount|total)"
        start_parsing = False
        
        # Simple row pattern: Description ... Qty ... Price ... Amount
        # e.g. "Widget A   2   10.00   20.00"
        # We look for lines ending with a number (Amount)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if re.search(header_pattern, line, re.IGNORECASE):
                start_parsing = True
                continue
            
            if start_parsing:
                # Stop if we hit totals
                if re.match(r"(subtotal|total|tax|amount due)", line, re.IGNORECASE):
                    break
                
                # Try to parse line
                # Look for at least a price at the end
                # Regex: (Description) (Qty)? (Price) (Total)
                # This is hard without layout, but let's try to match the last 2-3 numbers
                
                # Heuristic: Match "Description ... float ... float"
                # Group 1: Description, Group 2: Qty (Optional), Group 3: Unit Price, Group 4: Total
                # Very permissive regex
                
                row_reg = r"^(.+?)\s+(\d+)?\s*[\$]?\s*([0-9,]+\.\d{2})\s*[\$]?\s*([0-9,]+\.\d{2})$"
                match = re.search(row_reg, line)
                if match:
                    desc = match.group(1).strip()
                    qty = match.group(2) if match.group(2) else "1"
                    price = match.group(3)
                    total = match.group(4)
                    
                    items.append({
                        "description": self._create_field(desc, "string", 0.8, "regex_row"),
                        "quantity": self._create_field(qty, "number", 0.8, "regex_row"),
                        "unit_price": self._create_field(price, "currency", 0.8, "regex_row"),
                        "total_amount": self._create_field(total, "currency", 0.8, "regex_row")
                    })
        
        return items

    def extract_totals(self, text: str, line_items: List) -> Dict[str, Any]:
        """
        Extract numeric totals (Subtotal, Tax, Total) and validate.
        """
        totals = {}
        errors = []
        
        amounts = self.regex_extractor.extract_amounts(text)
        # Fallback if no specific labels found, usually biggest amount is Total
        
        # Look for labelled amounts using Regex
        # e.g. "Total: $500.00"
        
        # Update regex to be tolerant of 'Tax (10%)' where 10 contains digits.
        # We allow any characters (except colon or newline) ungreedily until the separator.
        label_patterns = {
            "subtotal": r"(?i)(subtotal|sub text|net amount|sub-total).*?[:\s]+[\$]?([0-9,]+\.\d{2})",
            "tax": r"(?i)(tax|vat|gst).*?[:\s]+[\$]?([0-9,]+\.\d{2})",
            "total_amount": r"(?i)\b(total|amount due|grand total)\b.*?[:\s]+[\$]?([0-9,]+\.\d{2})",
            "discount": r"(?i)(discount).*?[:\s]+[\$]?([0-9,]+\.\d{2})",
            "shipping": r"(?i)(shipping|freight).*?[:\s]+[\$]?([0-9,]+\.\d{2})"
        }
        
        for key, pat in label_patterns.items():
            matches = re.search(pat, text)
            if matches:
                val_str = matches.group(2).replace(',', '')
                try:
                    val_float = float(val_str)
                    totals[key] = self._create_field(val_float, "currency", 0.95, "regex_label")
                except ValueError:
                    pass
        
        # If total not found, try max amount
        if "total_amount" not in totals and amounts:
            # simple heuristic: max value
            max_val = -1.0
            best_item = None
            for item in amounts:
                if isinstance(item['value'], (int, float)):
                    if item['value'] > max_val:
                        max_val = item['value']
                        best_item = item
            if best_item:
                totals["total_amount"] = best_item
        
        # Validation: Sum of line items should ~ Subtotal
        # or Subtotal + Tax + Shipping - Discount = Total
        
        calc_subtotal = 0.0
        for item in line_items:
            try:
                # Handle potentially string or float content in 'value'
                t_val = item.get("total_amount", {}).get("value", 0)
                if isinstance(t_val, str):
                     t_val = float(t_val.replace(',', '').replace('$',''))
                calc_subtotal += float(t_val)
            except (ValueError, TypeError):
                pass
        
        extracted_subtotal = totals.get("subtotal", {}).get("value")
        if extracted_subtotal:
             # Check consistency
             diff = abs(calc_subtotal - float(extracted_subtotal))
             if diff > 1.0: # Allow small rounding diff
                 errors.append(f"Line items total ({calc_subtotal}) does not match extracted subtotal ({extracted_subtotal})")
        
        totals["validation_errors"] = errors
        
        return totals

    def extract_payment_terms(self, text: str) -> Dict[str, Any]:
        """
        Extract headers like 'Net 30', 'Due in 15 days'.
        """
        terms = {}
        patterns = [
            r"(net\s*\d+)",
            r"(due\s*in\s*\d+\s*days)",
            r"(payment\s*terms\s*[:]?\s*[\w\s]+)"
        ]
        
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                terms["term_description"] = self._create_field(
                    match.group(1), "string", 0.9, "regex"
                )
                break
                
        return terms
