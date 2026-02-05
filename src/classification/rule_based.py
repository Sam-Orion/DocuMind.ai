import logging
import re
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleBasedClassifier:
    """
    Classifies documents based on keyword matching and simple heuristic rules.
    """

    def __init__(self):
        """
        Initialize the rule-based classifier with keyword patterns for supported document types.
        """
        self.rules = {
            "Invoice": [
                r"\binvoice\b", r"\bbill to\b", r"\bdue date\b", r"\bbalance due\b", 
                r"\bsubtotal\b", r"\btax rate\b", r"\binvoice no\b", r"\binvoice number\b",
                r"\bgrand total\b", r"\bpayment terms\b"
            ],
            "Receipt": [
                r"\breceipt\b", r"\btransaction\b", r"\bthank you\b", r"\bcashier\b", 
                r"\bchange\b", r"\btotal amount\b", r"\bcard type\b", r"\bauth code\b",
                r"\btax invoice\b", r"\bpos\b"
            ],
            "Resume": [
                r"\bresume\b", r"\bcurriculum vitae\b", r"\bcv\b", r"\bexperience\b", 
                r"\beducation\b", r"\bskills\b", r"\bwork history\b", r"\bprojects\b", 
                r"\blanguages\b", r"\bcertifications\b", r"\bachievements\b"
            ],
            "ID Document": [
                r"\bpassport\b", r"\bdriver license\b", r"\bdriving licence\b", r"\bidentity card\b", 
                r"\bdate of birth\b", r"\bdob\b", r"\bnationality\b", r"\bsex\b", r"\bgender\b",
                r"\bissued on\b", r"\bexpiry date\b"
            ],
            "Business Card": [
                r"\btel\b", r"\bmobile\b", r"\bphone\b", r"\bemail\b", r"\bwebsite\b", 
                r"\bwww\b", r"\bfax\b", r"\bco\.", r"\bltd\.", r"\binc\."
            ]
        }
        logger.info("RuleBasedClassifier initialized with 5 document categories.")

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classifies the given text into one of the document categories.

        Args:
            text: The full text content of the document.

        Returns:
            Dict containing:
            - 'document_type': The predicted category.
            - 'confidence': Confidence score (0.0 to 1.0).
            - 'matched_rules': List of keywords that triggered the classification.
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for classification.")
            return {
                "document_type": "Unknown",
                "confidence": 0.0,
                "matched_rules": []
            }

        text_lower = text.lower()
        scores = {category: 0 for category in self.rules}
        matches = {category: [] for category in self.rules}

        # Calculate scores based on keyword matches
        for category, patterns in self.rules.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    scores[category] += 1
                    matches[category].append(pattern.replace(r"\b", "").replace("\\", ""))

        # Heuristic adjustments
        text_length = len(text)
        
        # Receipt/Business Card heuristics: Boost if text is short
        if text_length < 300:
             # Boost Business Card score lightly if short
            if scores["Business Card"] > 0:
                scores["Business Card"] += 1
            # Boost Receipt score if short and has minimal matches
            if scores["Receipt"] > 0:
                 scores["Receipt"] += 0.5
        
        # Resume heuristic: Usually long with many matches
        if text_length > 1000 and scores["Resume"] > 2:
            scores["Resume"] += 2

        # Find best match
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]
        total_possible_rules = len(self.rules[best_category])
        
        # Normalize confidence (clamped between 0 and 1)
        # Adding a base confidence if there is at least one strong match
        if best_score == 0:
            confidence = 0.0
            best_category = "Unknown"
        else:
            # Simple heuristic: score matches relative to a "strong" match threshold (e.g., 3-4 keywords)
            # We don't necessarily need ALL keywords to be 100% confident.
            confidence = min(best_score / 4.0, 1.0)
            
            # Reduce confidence slightly if it's very close to another category (ambiguity check could be added here)

        logger.info(f"Classified document as '{best_category}' with confidence {confidence:.2f}")

        return {
            "document_type": best_category,
            "confidence": float(confidence),
            "matched_rules": matches[best_category] if best_category != "Unknown" else []
        }
