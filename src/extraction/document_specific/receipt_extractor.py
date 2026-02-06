import re
import logging
from typing import Dict, Any, List, Optional
from src.extraction.base import BaseExtractor
from src.extraction.regex_extractor import RegexExtractor
from src.extraction.spacy_extractor import SpacyExtractor

logger = logging.getLogger(__name__)

class ReceiptExtractor(BaseExtractor):
    """
    Specialized extractor for Receipts.
    Extracts merchant details, transaction info, items, payment methods, and loyalty info.
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
        Extract all receipt fields from text.
        """
        if not text:
            logger.warning("Empty text provided to ReceiptExtractor")
            return {}

        logger.info("Starting receipt extraction")
        
        # 1. Merchant Information
        merchant = self.extract_merchant_info(text)
        
        # 2. Transaction Details
        transaction = self.extract_transaction_details(text)
        
        # 3. Items
        items = self.extract_items(text)
        
        # 4. Payment Info
        payment = self.extract_payment_info(text)
        
        # 5. Loyalty Info
        loyalty = self.extract_loyalty_info(text)
        
        # Calculate total from items or find explicit total
        # (Reusing some logic from invoice extractor might be good effectively, but keeping separate for now)
        total_amount = self._find_total(text, items)
        
        result = {
            "merchant": merchant,
            "transaction": transaction,
            "items": items,
            "payment": payment,
            "loyalty": loyalty,
            "total_amount": total_amount,
            "document_type": "receipt"
        }
        
        return result

    def extract_merchant_info(self, text: str) -> Dict[str, Any]:
        """
        Extract merchant name and address. Similar to Invoice but usually at the very top.
        """
        merchant_data = {}
        lines = text.split('\n')
        
        # Heuristic 1: The first non-empty line is often the Merchant Name
        for line in lines[:5]: # Check first 5 lines
            line = line.strip()
            if len(line) > 2: # Ignore unlikely short noise
                # If we have Spacy, verify it's ORG?
                if self.spacy_extractor:
                    # extract_entities returns Dict[str, List[Dict]]
                    # e.g., {'ORG': [{'value': 'SuperMart', ...}], 'PERSON': []}
                    ents_dict = self.spacy_extractor.extract_entities(line)
                    
                    # Check if 'ORG' key exists and has items
                    if ents_dict.get("ORG"):
                         merchant_data["name"] = self._create_field(line, "string", 0.8, "heuristic_top_line")
                         break
                
                # Fallback: Just take the first significant line if not found yet
                if "name" not in merchant_data:
                    merchant_data["name"] = self._create_field(line, "string", 0.6, "heuristic_first_line")
                    break
        
        # Address
        addresses = self.spacy_extractor.extract_addresses(text) if self.spacy_extractor else []
        if addresses:
            merchant_data["address"] = addresses[0]
            
        # Phone
        phones = self.regex_extractor.extract_phone_numbers(text)
        if phones:
            merchant_data["phone"] = phones[0]
            
        return merchant_data

    def extract_transaction_details(self, text: str) -> Dict[str, Any]:
        """
        Extract Date, Time, Receipt Number.
        """
        details = {}
        
        # Date
        dates = self.regex_extractor.extract_dates(text)
        if dates:
            details["date"] = dates[0]
            
        # Time (Simple regex for HH:MM:SS or HH:MM am/pm)
        time_pat = r"(\d{1,2}:\d{2}(?::\d{2})?\s*(?:[aApP][mM])?)"
        time_match = re.search(time_pat, text)
        if time_match:
            details["time"] = self._create_field(time_match.group(1), "time", 0.9, "regex")
            
        # Receipt / Transaction #
        # Often labeled "Rcpt#", "Trans#", "Invoice#", "Order#"
        # Use [ \t] to avoid matching across newlines (e.g. "receipt.\nThank")
        id_pat = r"(?i)\b(?:Rcpt|Receipt|Trans|Transaction|Trx|Order|Inv|Invoice)\b[ \t]*[:#.]+[ \t]*([A-Z0-9-]{4,})"
        id_match = re.search(id_pat, text)
        if id_match:
             details["receipt_number"] = self._create_field(id_match.group(1), "string", 0.9, "regex")
             
        # Terminal ID
        term_pat = r"(?i)\b(?:Term|Terminal)\b[ \t]*[:#.]+[ \t]*([A-Z0-9]+)"
        term_match = re.search(term_pat, text)
        if term_match:
            details["terminal_id"] = self._create_field(term_match.group(1), "string", 0.85, "regex")
            
        return details

    def extract_items(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract line items. Receipts often have: "Item Name  price" or "Item Name  qty  price"
        """
        items = []
        lines = text.split('\n')
        
        # Exclude common totals/payment lines to avoid false positives
        exclude_keywords = ["total", "subtotal", "tax", "change", "cash", "visa", "mastercard", "amex", "balance", "item"]
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
                
            # Skip if contains exclude keywords (case insensitive)
            if any(k in line_str.lower() for k in exclude_keywords):
                continue
                
            # Pattern 1: Text followed explicitly by a price at end
            # "Milk 2.99" or "Bread LB 2.49"
            # We want to be careful not to match dates or phone numbers
            
            # Simple heuristic: Ends with a float like structure
            price_pat = r"^(.+?)\s+[\$]?([0-9]+\.\d{2})\s*([T]?[A-Z]?)?$" 
            # Group 1: Desc, Group 2: Price, Group 3: Optional Tax Flag (T, F, etc)
            
            match = re.search(price_pat, line_str)
            if match:
                desc = match.group(1).strip()
                price = match.group(2)
                
                # Filter out likely non-items (too short, or numbers only)
                if len(desc) < 2 or desc.replace(" ", "").isdigit():
                    continue

                items.append({
                    "description": self._create_field(desc, "string", 0.7, "regex_line"),
                    "total_price": self._create_field(price, "currency", 0.7, "regex_line"),
                    "quantity": self._create_field("1", "number", 0.5, "default") # Assumed 1 unless found otherwise
                })
                
        return items

    def extract_payment_info(self, text: str) -> Dict[str, Any]:
        """
        Extract payment method, card info, etc.
        """
        payment = {}
        
        # Payment Method
        methods = ["Visa", "MasterCard", "Amex", "American Express", "Discover", "Cash", "Credit Card", "Debit Card"]
        found_method = None
        for m in methods:
            if re.search(r"\b" + re.escape(m) + r"\b", text, re.IGNORECASE):
                found_method = m
                break
        
        if found_method:
            payment["method"] = self._create_field(found_method, "string", 0.9, "keyword_search")
            
        # Card Last 4
        # "***********1234" or "Acct: ... 1234"
        last4_pat = r"(?:Acct|Card|Ends)\s*[:#]?\s*[\*xX\.]+(\d{4})"
        match = re.search(last4_pat, text, re.IGNORECASE)
        if match:
             payment["card_last_4"] = self._create_field(match.group(1), "string", 0.95, "regex")
             
        # Auth Code
        auth_pat = r"(?i)\b(?:Auth|Approval)(?:[ \t]*Code)?\b[ \t]*[:#.]+[ \t]*([A-Z0-9]+)"
        auth_match = re.search(auth_pat, text)
        if auth_match:
            payment["auth_code"] = self._create_field(auth_match.group(1), "string", 0.9, "regex")
            
        return payment

    def extract_loyalty_info(self, text: str) -> Dict[str, Any]:
        """
        Extract rewards info.
        """
        loyalty = {}
        
        # Member number
        mem_pat = r"(?:Member|Rewards)\s*(?:ID|#)?\s*[:]?\s*([0-9\-\s]{5,})" 
        match = re.search(mem_pat, text, re.IGNORECASE)
        if match:
            # clean up spaces/dashes if strictly number
            val = match.group(1).strip()
            loyalty["member_id"] = self._create_field(val, "string", 0.8, "regex")
            
        # Points Balance
        pts_pat = r"(?:Points|Balance)\s*[:]?\s*(\d+)"
        pts_match = re.search(pts_pat, text, re.IGNORECASE)
        if pts_match:
            loyalty["points_balance"] = self._create_field(pts_match.group(1), "number", 0.85, "regex")
            
        return loyalty

    def _find_total(self, text: str, items: List) -> Dict[str, Any]:
        """
        Extract Total Amount explicitly.
        """
        total_pat = r"(?i)\b(total|amount due|grand total)\b.*?[:\s]+[\$]?([0-9,]+\.\d{2})"
        matches = re.findall(total_pat, text)
        if matches:
            # Usually the last total on the receipt is the grand total
            val_str = matches[-1][1].replace(',', '')
            try:
                val = float(val_str)
                return self._create_field(val, "currency", 0.95, "regex_label")
            except ValueError:
                pass
        return {}
