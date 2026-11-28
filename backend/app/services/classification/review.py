"""Human review rules and explainability reasons (frozen per plan v2, section 8)."""

from typing import Final

# Thresholds (Section 8)
REVIEW_LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.70
REVIEW_LOW_MARGIN_THRESHOLD: Final[float] = 0.20
CONFLICT_MIN_PROBABILITY: Final[float] = 0.60

OTHER_DISASTER_CLASSES_IMAGE: Final[tuple[str, ...]] = ("earthquake", "landslide", "other_disaster")
OTHER_DISASTER_CLASSES_TEXT: Final[tuple[str, ...]] = ("earthquake",)

# All review reason codes in evaluation order (Section 8)
REVIEW_REASONS: Final[tuple[str, ...]] = (
    "NO_SUPPORTED_LABEL",
    "PREDICTED_OTHER_DISASTER",
    "PREDICTED_NOT_DISASTER",
    "LOW_CONFIDENCE",
    "LOW_MARGIN",
    "USER_LABEL_DISAGREES",
    "TEXT_IMAGE_CONFLICT",
    "IMAGE_NEEDS_REVIEW",
    "MODELS_UNAVAILABLE",
)


def evaluate_review_rules(
    final_label: str,
    top1: str,
    model_confidence: float | None,
    margin: float | None,
    user_label: str | None,
    text_top1_raw: str | None,
    text_top1_collapsed: str | None,
    text_top1_prob: float | None,
    image_top1_raw: str | None,
    image_top1_collapsed: str | None,
    image_top1_prob: float | None,
    any_image_needs_review: bool,
    ml_models_available: bool,
) -> tuple[bool, list[str]]:
    """Apply frozen human review rules (Section 8).

    Returns (human_review_required, list_of_firing_reasons).
    """
    reasons: list[str] = []

    # 1. NO_SUPPORTED_LABEL: final_label = UNDETERMINED
    if final_label == "UNDETERMINED":
        reasons.append("NO_SUPPORTED_LABEL")

    # 2. PREDICTED_OTHER_DISASTER:
    # OTHER won and (image top-1 in {earthquake, landslide, other_disaster} OR text top-1 is Earthquake)
    predicted_other_fired = False
    if top1 == "OTHER":
        img_other = (
            image_top1_raw is not None
            and image_top1_raw.strip().lower() in OTHER_DISASTER_CLASSES_IMAGE
        )
        txt_other = (
            text_top1_raw is not None
            and text_top1_raw.strip().lower() in OTHER_DISASTER_CLASSES_TEXT
        )
        if img_other or txt_other:
            reasons.append("PREDICTED_OTHER_DISASTER")
            predicted_other_fired = True

    # 3. PREDICTED_NOT_DISASTER:
    # OTHER won and the previous rule did not apply
    if top1 == "OTHER" and not predicted_other_fired:
        reasons.append("PREDICTED_NOT_DISASTER")

    # 4. LOW_CONFIDENCE: model_confidence < 0.70
    if model_confidence is not None and model_confidence < REVIEW_LOW_CONFIDENCE_THRESHOLD:
        reasons.append("LOW_CONFIDENCE")

    # 5. LOW_MARGIN: margin < 0.20
    if margin is not None and margin < REVIEW_LOW_MARGIN_THRESHOLD:
        reasons.append("LOW_MARGIN")

    # 6. USER_LABEL_DISAGREES: the citizen's mapped label != final_label
    if user_label is not None and user_label != final_label:
        reasons.append("USER_LABEL_DISAGREES")

    # 7. TEXT_IMAGE_CONFLICT:
    # collapsed text top-1 != collapsed image top-1, and both top-1 probabilities >= 0.60
    if (
        text_top1_collapsed is not None
        and image_top1_collapsed is not None
        and text_top1_collapsed != image_top1_collapsed
        and text_top1_prob is not None
        and text_top1_prob >= CONFLICT_MIN_PROBABILITY
        and image_top1_prob is not None
        and image_top1_prob >= CONFLICT_MIN_PROBABILITY
    ):
        reasons.append("TEXT_IMAGE_CONFLICT")

    # 8. IMAGE_NEEDS_REVIEW: any photo returned needs_review = true
    if any_image_needs_review:
        reasons.append("IMAGE_NEEDS_REVIEW")

    # 9. MODELS_UNAVAILABLE: no ML source succeeded
    if not ml_models_available:
        reasons.append("MODELS_UNAVAILABLE")

    return (len(reasons) > 0, reasons)
