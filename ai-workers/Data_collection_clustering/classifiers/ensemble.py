"""
Ensemble Disaster Classifier.
Coordinates Step 4: Executes image classifier on visual media and text classifier on textual reports.
"""

import logging
from typing import Optional
from collectors.models import Report
from classifiers.base import BaseTextClassifier, BaseImageClassifier
from classifiers.text_classifier import DefaultTextClassifier
from classifiers.image_classifier import DefaultImageClassifier

logger = logging.getLogger("DisasterHub.EnsembleClassifier")

class DisasterEnsembleClassifier:
    """Manages inference across text and image models and fuses the multi-modal outputs."""

    def __init__(self, text_classifier: Optional[BaseTextClassifier] = None,
                 image_classifier: Optional[BaseImageClassifier] = None):
        self.text_classifier = text_classifier or DefaultTextClassifier()
        self.image_classifier = image_classifier or DefaultImageClassifier()

    def set_text_classifier(self, custom_classifier: BaseTextClassifier):
        """Allows user to plug in their custom trained text model."""
        self.text_classifier = custom_classifier
        logger.info(f"Custom text classifier registered: {type(custom_classifier).__name__}")

    def set_image_classifier(self, custom_classifier: BaseImageClassifier):
        """Allows user to plug in their custom trained vision model."""
        self.image_classifier = custom_classifier
        logger.info(f"Custom image classifier registered: {type(custom_classifier).__name__}")

    def classify_report(self, report: Report) -> Report:
        """Run multi-modal classification on the report."""
        # 1. Text Classification
        text_payload = f"{report.title}. {report.text}"
        text_res = self.text_classifier.predict(text_payload)
        report.text_classifier_label = text_res.label
        report.text_classifier_confidence = text_res.confidence

        # 2. Image Classification (prefers local downloaded file, falls back to URL)
        image_target = None
        if report.local_image_paths:
            image_target = report.local_image_paths[0]
        elif report.media_urls:
            image_target = report.media_urls[0]

        if image_target:
            img_res = self.image_classifier.predict(image_target)
            report.image_classifier_label = img_res.label
            report.image_classifier_confidence = img_res.confidence
        else:
            report.image_classifier_label = "No_Image"
            report.image_classifier_confidence = 0.0

        # 3. Decision Fusion Logic
        if report.image_classifier_label not in ["No_Image", "Disaster_Visual_Evidence"] and report.image_classifier_confidence >= 0.75:
            if report.image_classifier_label == report.text_classifier_label:
                report.ensemble_disaster_type = report.text_classifier_label
            else:
                report.ensemble_disaster_type = report.text_classifier_label
        else:
            report.ensemble_disaster_type = report.text_classifier_label

        report.disaster_type = report.ensemble_disaster_type
        return report
