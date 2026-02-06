from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    """
    Abstract Base Class for all document extractors.
    Enforces a standard interface and provides common utility methods.
    """

    def __init__(self):
        pass

    @abstractmethod
    def extract(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Main extraction method that must be implemented by all subclasses.
        
        Args:
            text: The raw text extracted from the document (OCR output).
            **kwargs: Additional arguments (e.g., image dimensions, layout data).
            
        Returns:
            A dictionary containing all extracted fields with confidence scores.
        """
        pass

    def _normalize_confidence(self, score: float) -> float:
        """
        Ensure confidence score is between 0.0 and 1.0.
        """
        return max(0.0, min(1.0, float(score)))

    def _create_field(self, value: Any, field_type: str, confidence: float = 1.0, 
                      source: str = "unknown", position: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Helper to create a standardized field dictionary.
        """
        return {
            "value": value,
            "field_type": field_type,
            "confidence": self._normalize_confidence(confidence),
            "source": source,
            "position": position
        }
