"""
Text Classifier Implementation for Disaster Hub.
Provides baseline NLP classification with probabilistic confidence scoring.
Easily swapped when user provides a trained model (BERT, RoBERTa, FastText, etc.).
"""

import math
import re
from typing import Dict, List
from classifiers.base import BaseTextClassifier, ClassificationResult

DISASTER_FEATURE_WEIGHTS = {
    "Flood": {
        "flood": 4.0, "waterlogging": 3.5, "submerged": 3.5, "inundated": 3.5,
        "overflowing": 2.5, "cusecs": 3.0, "dam": 2.0, "drowning": 2.5,
        "river danger mark": 3.0, "water level": 2.0, "rescue boat": 2.5
    },
    "Cyclone": {
        "cyclone": 4.5, "storm": 2.5, "gale": 3.5, "gusts": 3.0, "landfall": 4.0,
        "depression": 2.5, "squally": 3.0, "tidal wave": 3.5, "coastal winds": 3.0,
        "trees uprooted": 2.5, "red alert": 2.0
    },
    "Fire": {
        "fire": 4.0, "blaze": 4.0, "inferno": 3.5, "chemical": 2.5, "explosion": 3.5,
        "smoke": 2.0, "fire tenders": 3.5, "fire brigade": 3.5, "burning": 2.5
    },
    "Earthquake": {
        "earthquake": 4.5, "tremor": 4.0, "seismic": 3.5, "richter": 4.0,
        "epicenter": 3.5, "magnitude": 3.5, "aftershocks": 3.5
    },
    "Structural_Collapse": {
        "collapse": 4.0, "debris": 3.5, "rubble": 3.5, "trapped": 3.0,
        "building fall": 4.0, "wall fall": 3.5, "slab collapse": 4.0
    },
    "Landslide": {
        "landslide": 4.5, "mudslide": 4.0, "rockfall": 3.5, "debris flow": 3.0
    },
    "Heatwave": {
        "heatwave": 4.5, "extreme heat": 3.5, "sunstroke": 4.0, "high temperature": 2.5
    },
    "Drought": {
        "drought": 4.5, "scarcity": 3.0, "dry spell": 3.5, "parched": 3.0
    }
}

class DefaultTextClassifier(BaseTextClassifier):
    """Rule and keyword-weighted baseline text classifier with softmax probabilities."""

    def __init__(self, model_name: str = "DisasterNLP-RuleBaseline-v1"):
        self.model_name = model_name
        self.feature_weights = DISASTER_FEATURE_WEIGHTS

    def predict(self, text: str) -> ClassificationResult:
        """Classify report text into a disaster type with normalized confidence."""
        text_lower = text.lower()
        raw_scores: Dict[str, float] = {k: 0.1 for k in self.feature_weights}

        for disaster, keywords in self.feature_weights.items():
            for kw, weight in keywords.items():
                if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                    raw_scores[disaster] += weight

        # Softmax normalization to get probabilities
        max_score = max(raw_scores.values())
        if max_score <= 0.1:
            # No strong cues, assign fallback
            return ClassificationResult(
                label="Other_Emergency",
                confidence=0.40,
                probabilities={k: 1.0 / len(raw_scores) for k in raw_scores},
                model_name=self.model_name,
                metadata={"status": "low_confidence_fallback"}
            )

        exp_scores = {k: math.exp(v) for k, v in raw_scores.items()}
        sum_exp = sum(exp_scores.values())
        probabilities = {k: v / sum_exp for k, v in exp_scores.items()}

        top_label = max(probabilities, key=probabilities.get)
        confidence = probabilities[top_label]

        return ClassificationResult(
            label=top_label,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in probabilities.items()},
            model_name=self.model_name,
            metadata={"top_features_matched": [kw for kw in self.feature_weights[top_label] if kw in text_lower]}
        )
