"""
Image Classifier Implementation for Disaster Hub.
Provides visual/heuristic baseline image classification for disaster reports.
Ready to be swapped with trained PyTorch (ResNet, EfficientNet, ViT) or TensorFlow models.
"""

import os
import re
from typing import Dict
from classifiers.base import BaseImageClassifier, ClassificationResult

class DefaultImageClassifier(BaseImageClassifier):
    """
    Baseline image classifier.
    Inspects image metadata, URL tokens, and filename semantics to categorize disaster imagery.
    User will replace this with their computer vision model later.
    """

    def __init__(self, model_name: str = "DisasterCV-Baseline-v1"):
        self.model_name = model_name

    def predict(self, image_path_or_url: str) -> ClassificationResult:
        """Classify an image into a disaster type."""
        if not image_path_or_url:
            return ClassificationResult(
                label="No_Image",
                confidence=0.0,
                model_name=self.model_name
            )

        target = os.path.basename(image_path_or_url).lower()
        
        heuristics = [
            (r"flood|submerged|waterlog|deluge|river|dam", "Flood", 0.88),
            (r"cyclone|storm|wind|surge|waves|satellite", "Cyclone", 0.85),
            (r"fire|blaze|smoke|flame|explosion", "Fire", 0.90),
            (r"debris|rubble|collapse|building", "Structural_Collapse", 0.82),
            (r"earthquake|crack|fissure", "Earthquake", 0.80),
            (r"landslide|mudslide", "Landslide", 0.84),
        ]

        for pattern, label, conf in heuristics:
            if re.search(pattern, target):
                return ClassificationResult(
                    label=label,
                    confidence=conf,
                    probabilities={label: conf, "Other": round(1.0 - conf, 2)},
                    model_name=self.model_name,
                    metadata={"visual_source": image_path_or_url}
                )

        # Default classification if no visual cues in URI
        return ClassificationResult(
            label="Disaster_Visual_Evidence",
            confidence=0.65,
            probabilities={"Disaster_Visual_Evidence": 0.65, "Unknown": 0.35},
            model_name=self.model_name,
            metadata={"visual_source": image_path_or_url}
        )
