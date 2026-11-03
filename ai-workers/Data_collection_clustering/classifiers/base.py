"""
Abstract Base Classes for Text and Image Classifiers.
Allows seamless plug-and-play when custom PyTorch / TensorFlow / HuggingFace models are provided.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class ClassificationResult:
    """Standardized output from any classifier."""
    label: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    model_name: str = "BaseModel"
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseTextClassifier(ABC):
    """Abstract interface for text classifiers."""

    @abstractmethod
    def predict(self, text: str) -> ClassificationResult:
        """
        Classify a single text string into a disaster type.
        """
        pass

    def predict_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify a batch of text strings. Can be overridden for GPU batched inference.
        """
        return [self.predict(t) for t in texts]

class BaseImageClassifier(ABC):
    """Abstract interface for image classifiers."""

    @abstractmethod
    def predict(self, image_path_or_url: str) -> ClassificationResult:
        """
        Classify a single image (from local path or remote URL) into a disaster type.
        """
        pass

    def predict_batch(self, image_paths_or_urls: List[str]) -> List[ClassificationResult]:
        """
        Classify a batch of images. Can be overridden for GPU batched inference.
        """
        return [self.predict(p) for p in image_paths_or_urls]
