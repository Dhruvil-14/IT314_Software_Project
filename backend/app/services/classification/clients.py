"""HTTP clients for deployed text (IndicBERT) and image (ConvNeXt) classifiers."""

import logging
import math
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import get_settings
from app.services.ai_provider import scrub_pii
from app.services.classification.fusion import get_top1_with_tiebreak
from app.services.classification.labels import (
    collapse_image_probabilities,
    collapse_text_probabilities,
)

logger = logging.getLogger("drch.classification.clients")

DEFAULT_TIMEOUT_SECONDS = 15.0


class ClassifierClientError(Exception):
    """Base error for classification client failures."""


class ClassifierValidationError(ClassifierClientError):
    """Raised when model returns malformed, unnormalized, or invalid JSON payload."""


@dataclass(frozen=True)
class TextPredictionResult:
    raw_prediction: str
    confidence: float
    raw_probabilities: dict[str, float]
    collapsed_probabilities: dict[str, float]
    collapsed_top1: str
    collapsed_top1_prob: float


@dataclass(frozen=True)
class ImagePredictionResult:
    evidence_id: str | None
    filename: str
    raw_prediction: str
    confidence: float
    needs_review: bool
    all_probabilities: dict[str, float]
    collapsed_probabilities: dict[str, float]
    collapsed_top1: str
    collapsed_top1_prob: float


def _validate_probabilities(probs: Any, sum_tolerance: float = 0.02) -> dict[str, float]:
    """Validate probability dictionary: numeric, finite, in [0, 1], and sum to 1.0 ± tolerance."""
    if not isinstance(probs, dict) or not probs:
        raise ClassifierValidationError("Missing or invalid probabilities dictionary")

    cleaned: dict[str, float] = {}
    for key, val in probs.items():
        if isinstance(val, bool):
            raise ClassifierValidationError(f"Invalid boolean probability for key {key}")
        try:
            fval = float(val)
        except (TypeError, ValueError) as exc:
            raise ClassifierValidationError(f"Non-numeric probability for key {key}") from exc

        if math.isnan(fval) or math.isinf(fval):
            raise ClassifierValidationError(f"NaN or Inf probability for key {key}")
        if not (-1e-6 <= fval <= 1.0 + 1e-6):
            raise ClassifierValidationError(f"Probability out of bounds [0, 1] for key {key}: {fval}")
        cleaned[str(key)] = max(0.0, min(1.0, fval))

    prob_sum = sum(cleaned.values())
    if abs(prob_sum - 1.0) > sum_tolerance:
        raise ClassifierValidationError(f"Probabilities do not sum to 1.0 ± {sum_tolerance} (sum={prob_sum:.4f})")

    return cleaned


def classify_text(
    description: str,
    endpoint_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> TextPredictionResult | None:
    """Call IndicBERT text classification API.

    Per Section 6:
    - Text sent: description after scrub_pii()
    - Never sent: name, phone, email, coordinates, address, landmark
    - URL: DISASTER_CLASSIFIER_URL is used as given (full /predict URL)
    """
    url = endpoint_url or get_settings().disaster_classifier_url
    if not url:
        logger.debug("Text classifier URL not configured; skipping text classification")
        return None

    # Scrub PII strictly before transmission
    scrubbed = str(scrub_pii(description)).strip()
    if not scrubbed:
        logger.warning("Description is empty after PII scrubbing; skipping text classification")
        return None

    try:
        response = httpx.post(
            url,
            json={"text": scrubbed},
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        if response.status_code != 200:
            logger.warning(
                "Text classifier returned non-200 status %d: %s",
                response.status_code,
                response.text[:200],
            )
            return None

        data = response.json()
        if not isinstance(data, dict):
            raise ClassifierValidationError("Response payload is not a JSON object")

        raw_pred = data.get("prediction")
        if not isinstance(raw_pred, str) or not raw_pred.strip():
            raise ClassifierValidationError("Missing or empty prediction string")

        raw_conf = data.get("confidence")
        if isinstance(raw_conf, bool) or raw_conf is None:
            raise ClassifierValidationError("Invalid confidence value")
        conf_float = float(raw_conf)
        if not 0.0 <= conf_float <= 1.0:
            raise ClassifierValidationError(f"Confidence out of bounds: {conf_float}")

        raw_probs = _validate_probabilities(data.get("probabilities"))
        collapsed = collapse_text_probabilities(raw_probs)
        top1_col = get_top1_with_tiebreak(collapsed)

        return TextPredictionResult(
            raw_prediction=raw_pred.strip(),
            confidence=conf_float,
            raw_probabilities=raw_probs,
            collapsed_probabilities=collapsed,
            collapsed_top1=top1_col,
            collapsed_top1_prob=collapsed[top1_col],
        )

    except (httpx.HTTPError, OSError) as exc:
        logger.warning("Text classifier request failed: %s (%s)", type(exc).__name__, exc)
        return None
    except ClassifierValidationError as exc:
        logger.warning("Text classifier validation failed: %s", exc)
        return None
    except Exception:
        logger.exception("Unexpected error during text classification")
        return None


def classify_image(
    image_bytes: bytes,
    filename: str = "evidence.jpg",
    content_type: str = "image/jpeg",
    evidence_id: str | None = None,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> ImagePredictionResult | None:
    """Call ConvNeXt image classification API.

    Per Section 6 & 13:
    - Input sent: each photo's stored bytes (EXIF already stripped)
    - Never sent: metadata, identity, location
    - URL: AI_IMAGE_SERVICE_URL is base URL; /predict is appended
    """
    configured_base = base_url or get_settings().ai_image_service_url
    if not configured_base:
        logger.debug("Image classifier base URL not configured; skipping image classification")
        return None

    predict_url = f"{configured_base.rstrip('/')}/predict"

    try:
        files = {"file": (filename, image_bytes, content_type)}
        response = httpx.post(
            predict_url,
            files=files,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        if response.status_code != 200:
            logger.warning(
                "Image classifier returned non-200 status %d: %s",
                response.status_code,
                response.text[:200],
            )
            return None

        data = response.json()
        if not isinstance(data, dict):
            raise ClassifierValidationError("Response payload is not a JSON object")

        raw_pred = data.get("prediction")
        if not isinstance(raw_pred, str) or not raw_pred.strip():
            raise ClassifierValidationError("Missing or empty prediction string")

        raw_conf = data.get("confidence")
        if isinstance(raw_conf, bool) or raw_conf is None:
            raise ClassifierValidationError("Invalid confidence value")
        conf_float = float(raw_conf)
        if not 0.0 <= conf_float <= 1.0:
            raise ClassifierValidationError(f"Confidence out of bounds: {conf_float}")

        needs_review = bool(data.get("needs_review", False))
        all_probs = _validate_probabilities(data.get("all_probabilities"))
        collapsed = collapse_image_probabilities(all_probs)
        top1_col = get_top1_with_tiebreak(collapsed)

        return ImagePredictionResult(
            evidence_id=evidence_id,
            filename=filename,
            raw_prediction=raw_pred.strip(),
            confidence=conf_float,
            needs_review=needs_review,
            all_probabilities=all_probs,
            collapsed_probabilities=collapsed,
            collapsed_top1=top1_col,
            collapsed_top1_prob=collapsed[top1_col],
        )

    except (httpx.HTTPError, OSError) as exc:
        logger.warning("Image classifier request failed: %s (%s)", type(exc).__name__, exc)
        return None
    except ClassifierValidationError as exc:
        logger.warning("Image classifier validation failed: %s", exc)
        return None
    except Exception:
        logger.exception("Unexpected error during image classification")
        return None
