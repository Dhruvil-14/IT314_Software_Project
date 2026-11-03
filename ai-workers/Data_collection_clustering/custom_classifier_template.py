"""
Template and Guide for Plugging In Your Custom Text and Image Classifiers.

When you have your trained text and image models, implement the simple
BaseTextClassifier and BaseImageClassifier interfaces below, then register
them with the pipeline:

    pipeline = DisasterHubPipeline()
    pipeline.set_text_classifier(MyCustomTextClassifier())
    pipeline.set_image_classifier(MyCustomImageClassifier())
    pipeline.run()
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifiers.base import BaseTextClassifier, BaseImageClassifier, ClassificationResult
from pipeline import DisasterHubPipeline

# ==============================================================================
# 1. TEMPLATE: CUSTOM TEXT CLASSIFIER (e.g. PyTorch / HuggingFace BERT / TF)
# ==============================================================================
class MyCustomTextClassifier(BaseTextClassifier):
    """
    Example user-supplied text classifier.
    Replace the logic inside predict() with your actual PyTorch/HuggingFace model inference:
    
    Example with HuggingFace:
        from transformers import pipeline
        self.hf_pipe = pipeline("text-classification", model="my-disaster-bert")
    """

    def __init__(self, model_weights_path: str = None):
        print(f"[Init] Loading custom text model from: {model_weights_path or 'In-memory weights'}")
        # e.g.: self.model = torch.load(model_weights_path)
        # e.g.: self.tokenizer = AutoTokenizer.from_pretrained(...)

    def predict(self, text: str) -> ClassificationResult:
        """
        Takes raw text and returns a ClassificationResult with:
          - label: e.g., 'Flood', 'Cyclone', 'Earthquake', 'Fire', 'Structural_Collapse'
          - confidence: float between 0.0 and 1.0
          - probabilities: optional dictionary of class probabilities
        """
        # --- REPLACE THIS DUMMY LOGIC WITH YOUR MODEL FORWARD PASS ---
        # inputs = self.tokenizer(text, return_tensors="pt")
        # logits = self.model(**inputs).logits
        # probs = torch.softmax(logits, dim=-1)
        
        # Example dummy output:
        predicted_label = "Flood" if "flood" in text.lower() or "water" in text.lower() else "Cyclone"
        confidence = 0.94

        return ClassificationResult(
            label=predicted_label,
            confidence=confidence,
            probabilities={predicted_label: confidence, "Other": round(1.0 - confidence, 2)},
            model_name="UserCustomBERTClassifier"
        )


# ==============================================================================
# 2. TEMPLATE: CUSTOM IMAGE CLASSIFIER (e.g. ResNet / EfficientNet / ViT)
# ==============================================================================
class MyCustomImageClassifier(BaseImageClassifier):
    """
    Example user-supplied vision classifier.
    Replace logic inside predict() with your PyTorch torchvision or HuggingFace ViT:
    
    Example with PyTorch:
        import torch
        from PIL import Image
        from torchvision import transforms
        self.model = torch.load('disaster_resnet50.pth').eval()
    """

    def __init__(self, model_weights_path: str = None):
        print(f"[Init] Loading custom vision model from: {model_weights_path or 'In-memory weights'}")
        # e.g.: self.model = torchvision.models.resnet50(pretrained=False)

    def predict(self, image_path_or_url: str) -> ClassificationResult:
        """
        Takes an image path or URL and returns a ClassificationResult.
        """
        # --- REPLACE THIS DUMMY LOGIC WITH YOUR CV MODEL INFERENCE ---
        # image = Image.open(image_path_or_url)
        # tensor = self.transform(image).unsqueeze(0)
        # outputs = self.model(tensor)
        
        # Example dummy output:
        predicted_label = "Flood"
        confidence = 0.91

        return ClassificationResult(
            label=predicted_label,
            confidence=confidence,
            probabilities={"Flood": 0.91, "Fire": 0.05, "Other": 0.04},
            model_name="UserCustomResNetClassifier"
        )


# ==============================================================================
# 3. HOW TO RUN THE COMPLETE PIPELINE WITH YOUR CUSTOM MODELS
# ==============================================================================
if __name__ == "__main__":
    print("Initializing pipeline with custom classifiers...")
    pipeline = DisasterHubPipeline()

    # Plug in custom models
    pipeline.set_text_classifier(MyCustomTextClassifier())
    pipeline.set_image_classifier(MyCustomImageClassifier())

    # Run complete Sprint 2 pipeline
    result = pipeline.run(use_live_feeds=False)  # set True for live online feeds
    print(f"Pipeline finished! Saved {result['cleaned_reports_count']} reports and {result['incidents_count']} incidents.")
