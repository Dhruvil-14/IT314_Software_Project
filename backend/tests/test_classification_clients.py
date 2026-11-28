"""Contract tests for IndicBERT text and ConvNeXt image classifier HTTP clients (Phase 2)."""

from unittest.mock import patch

import httpx
import pytest

from app.services.classification.clients import (
    _validate_probabilities,
    classify_image,
    classify_text,
)

# ═══════════════════════════════════════════════════════════════════════
# 1. Probability Validation Tests
# ═══════════════════════════════════════════════════════════════════════

def test_validate_probabilities_accepts_clean_dict():
    raw = {"Flood": 0.80, "Fire": 0.10, "Cyclone": 0.05, "OTHER": 0.05}
    cleaned = _validate_probabilities(raw)
    assert pytest.approx(sum(cleaned.values())) == 1.00


def test_validate_probabilities_rejects_bad_sum():
    from app.services.classification.clients import ClassifierValidationError

    with pytest.raises(ClassifierValidationError, match="do not sum to 1.0"):
        _validate_probabilities({"Flood": 0.80, "Fire": 0.10})  # sum 0.90 < 0.98


def test_validate_probabilities_rejects_nan_and_inf():
    from app.services.classification.clients import ClassifierValidationError

    with pytest.raises(ClassifierValidationError, match="NaN or Inf"):
        _validate_probabilities({"Flood": float("nan"), "Fire": 0.5})

    with pytest.raises(ClassifierValidationError, match="NaN or Inf"):
        _validate_probabilities({"Flood": float("inf"), "Fire": 0.5})


def test_validate_probabilities_rejects_negative_or_out_of_bounds():
    from app.services.classification.clients import ClassifierValidationError

    with pytest.raises(ClassifierValidationError, match="out of bounds"):
        _validate_probabilities({"Flood": -0.1, "Fire": 1.1})


# ═══════════════════════════════════════════════════════════════════════
# 2. Text Classifier Contract Tests (IndicBERT)
# ═══════════════════════════════════════════════════════════════════════

def test_classify_text_success_uses_full_url():
    """Verify text classifier uses DISASTER_CLASSIFIER_URL as given, scrubs PII, and returns collapsed result."""
    text_payload = {
        "prediction": "Flood",
        "confidence": 0.82,
        "probabilities": {
            "Flood": 0.82,
            "Fire": 0.03,
            "Cyclone": 0.08,
            "Earthquake": 0.02,
            "Non-Disaster": 0.05,
        },
    }

    mock_resp = httpx.Response(200, json=text_payload)
    with patch("httpx.post", return_value=mock_resp) as mock_post:
        res = classify_text(
            description="My phone is 9876543210 and water entering homes in Ellis Bridge",
            endpoint_url="https://it314-software-project-text-classifier.onrender.com/predict",
        )

        assert mock_post.called
        call_url, call_kwargs = mock_post.call_args
        assert call_url[0] == "https://it314-software-project-text-classifier.onrender.com/predict"
        # Confirm PII was scrubbed: phone number replaced
        sent_text = call_kwargs["json"]["text"]
        assert "9876543210" not in sent_text
        assert "[PHONE" in sent_text or "Ellis Bridge" in sent_text

        assert res is not None
        assert res.raw_prediction == "Flood"
        assert res.confidence == 0.82
        assert res.collapsed_top1 == "FLOOD"
        assert pytest.approx(res.collapsed_probabilities["FLOOD"]) == 0.82
        assert pytest.approx(res.collapsed_probabilities["OTHER"]) == 0.07


def test_classify_text_handles_non_200():
    mock_resp = httpx.Response(500, text="Internal Server Error")
    with patch("httpx.post", return_value=mock_resp):
        res = classify_text(
            description="Flooding near riverfront",
            endpoint_url="https://it314-software-project-text-classifier.onrender.com/predict",
        )
        assert res is None


def test_classify_text_handles_timeout():
    with patch("httpx.post", side_effect=httpx.ConnectTimeout("Connection timed out")):
        res = classify_text(
            description="Flooding near riverfront",
            endpoint_url="https://it314-software-project-text-classifier.onrender.com/predict",
        )
        assert res is None


def test_classify_text_handles_bad_probabilities():
    # Bad sum: sum is 0.50
    mock_resp = httpx.Response(200, json={"prediction": "Flood", "confidence": 0.5, "probabilities": {"Flood": 0.5}})
    with patch("httpx.post", return_value=mock_resp):
        res = classify_text(
            description="Flooding near riverfront",
            endpoint_url="https://it314-software-project-text-classifier.onrender.com/predict",
        )
        assert res is None


def test_classify_text_skips_when_url_missing():
    res = classify_text("Flooding near riverfront", endpoint_url=None)
    assert res is None


# ═══════════════════════════════════════════════════════════════════════
# 3. Image Classifier Contract Tests (ConvNeXt)
# ═══════════════════════════════════════════════════════════════════════

def test_classify_image_success_appends_predict():
    """Verify image classifier appends /predict to AI_IMAGE_SERVICE_URL and passes clean bytes."""
    image_payload = {
        "filename": "scene.jpg",
        "prediction": "flood",
        "confidence": 0.91,
        "needs_review": False,
        "all_probabilities": {
            "earthquake": 0.01,
            "fire": 0.01,
            "flood": 0.91,
            "hurricane": 0.04,
            "landslide": 0.01,
            "not_disaster": 0.01,
            "other_disaster": 0.01,
        },
    }

    mock_resp = httpx.Response(200, json=image_payload)
    with patch("httpx.post", return_value=mock_resp) as mock_post:
        res = classify_image(
            image_bytes=b"dummy_image_data",
            filename="scene.jpg",
            content_type="image/jpeg",
            evidence_id="123e4567-e89b-12d3-a456-426614174000",
            base_url="https://it314-software-project-dktp.onrender.com",
        )

        assert mock_post.called
        call_url, call_kwargs = mock_post.call_args
        # Confirms /predict was appended to base URL
        assert call_url[0] == "https://it314-software-project-dktp.onrender.com/predict"
        assert "files" in call_kwargs

        assert res is not None
        assert res.raw_prediction == "flood"
        assert res.confidence == 0.91
        assert not res.needs_review
        assert res.collapsed_top1 == "FLOOD"
        assert pytest.approx(res.collapsed_probabilities["FLOOD"]) == 0.91
        assert pytest.approx(res.collapsed_probabilities["CYCLONE"]) == 0.04
        assert pytest.approx(res.collapsed_probabilities["OTHER"]) == 0.04


def test_classify_image_handles_non_200():
    mock_resp = httpx.Response(503, text="Service Unavailable")
    with patch("httpx.post", return_value=mock_resp):
        res = classify_image(
            image_bytes=b"bytes",
            base_url="https://it314-software-project-dktp.onrender.com",
        )
        assert res is None


def test_classify_image_handles_timeout():
    with patch("httpx.post", side_effect=httpx.ReadTimeout("Read timed out")):
        res = classify_image(
            image_bytes=b"bytes",
            base_url="https://it314-software-project-dktp.onrender.com",
        )
        assert res is None


def test_classify_image_handles_malformed_probabilities():
    mock_resp = httpx.Response(200, json={"prediction": "flood", "confidence": 0.9, "all_probabilities": "not_a_dict"})
    with patch("httpx.post", return_value=mock_resp):
        res = classify_image(
            image_bytes=b"bytes",
            base_url="https://it314-software-project-dktp.onrender.com",
        )
        assert res is None


def test_classify_image_skips_when_url_missing():
    res = classify_image(b"bytes", base_url=None)
    assert res is None
