"""Weighted probability pooling and fusion logic (frozen per plan v2, section 7)."""

from dataclasses import dataclass
from typing import Any, Final

from app.services.classification.labels import (
    CLASSES,
    collapse_image_probabilities,
)

# Weights (Section 7.2)
WEIGHT_IMAGE: Final[float] = 0.45
WEIGHT_TEXT: Final[float] = 0.35
WEIGHT_CITIZEN: Final[float] = 0.20

# Citizen prior distribution (Section 7.1)
CITIZEN_PRIOR_CHOSEN: Final[float] = 0.70
CITIZEN_PRIOR_OTHER: Final[float] = 0.10

# Confidence level thresholds (Section 7.4)
CONFIDENCE_HIGH_THRESHOLD: Final[float] = 0.80
CONFIDENCE_MEDIUM_THRESHOLD: Final[float] = 0.60


@dataclass(frozen=True)
class FusionResult:
    fused_probabilities: dict[str, float]
    top1: str
    final_label: str
    model_confidence: float | None
    confidence_level: str | None
    margin: float
    agreement: str
    label_source: str


def build_citizen_vector(user_label: str) -> dict[str, float]:
    """Return citizen prior vector with 0.70 for chosen class and 0.10 for the other three (Section 7.1)."""
    return {c: CITIZEN_PRIOR_CHOSEN if c == user_label else CITIZEN_PRIOR_OTHER for c in CLASSES}


def aggregate_photo_vectors(photos: list[dict[str, Any]]) -> tuple[dict[str, float] | None, list[dict[str, Any]]]:
    """Aggregate per-photo ConvNeXt predictions per Section 7.1.

    1. keep photos whose top-1 class is not 'not_disaster'
    2. if none remain, keep all
    3. I = element-wise mean of the kept vectors
    """
    if not photos:
        return None, []

    # 1. Filter out photos whose top-1 class is 'not_disaster'
    kept = [p for p in photos if str(p.get("prediction", "")).strip().lower() != "not_disaster"]

    # 2. If none remain, keep all
    if not kept:
        kept = list(photos)

    # 3. Collapse and average
    collapsed_list = [collapse_image_probabilities(p.get("all_probabilities", {})) for p in kept]
    num_photos = len(collapsed_list)
    mean_vector: dict[str, float] = {
        c: sum(vec[c] for vec in collapsed_list) / num_photos for c in CLASSES
    }
    return mean_vector, kept


def get_top1_with_tiebreak(vector: dict[str, float]) -> str:
    """Return top-1 class using frozen tie-breaking order: FLOOD, FIRE, CYCLONE, OTHER (Section 7.3)."""
    return max(CLASSES, key=lambda c: (vector[c], -CLASSES.index(c)))


def fuse_sources(
    citizen_label: str | None,
    text_collapsed: dict[str, float] | None,
    image_collapsed: dict[str, float] | None,
) -> FusionResult:
    """Fuse present sources via weighted probability averaging (Section 7.3)."""
    present_weights: dict[str, float] = {}
    if image_collapsed is not None:
        present_weights["image"] = WEIGHT_IMAGE
    if text_collapsed is not None:
        present_weights["text"] = WEIGHT_TEXT
    if citizen_label is not None:
        present_weights["citizen"] = WEIGHT_CITIZEN

    weight_sum = sum(present_weights.values())
    if weight_sum <= 0:
        raise ValueError("Cannot perform fusion with zero sources present")

    # Weighted probability pooling (Section 7.3)
    p_vector: dict[str, float] = {c: 0.0 for c in CLASSES}
    if "image" in present_weights and image_collapsed is not None:
        w_img = present_weights["image"]
        for c in CLASSES:
            p_vector[c] += w_img * image_collapsed[c]

    if "text" in present_weights and text_collapsed is not None:
        w_txt = present_weights["text"]
        for c in CLASSES:
            p_vector[c] += w_txt * text_collapsed[c]

    if "citizen" in present_weights and citizen_label is not None:
        w_cit = present_weights["citizen"]
        cit_vec = build_citizen_vector(citizen_label)
        for c in CLASSES:
            p_vector[c] += w_cit * cit_vec[c]

    for c in CLASSES:
        p_vector[c] /= weight_sum

    top1 = get_top1_with_tiebreak(p_vector)
    final_label = "UNDETERMINED" if top1 == "OTHER" else top1

    has_ml = ("image" in present_weights) or ("text" in present_weights)
    model_confidence = p_vector[top1] if has_ml else None

    # Margin: top1 - second highest
    sorted_probs = sorted(p_vector.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]

    # Confidence level (Section 7.4)
    if model_confidence is None:
        confidence_level = None
    elif model_confidence >= CONFIDENCE_HIGH_THRESHOLD:
        confidence_level = "HIGH"
    elif model_confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    # Agreement (Section 7.5)
    source_top1s: list[str] = []
    if "citizen" in present_weights and citizen_label is not None:
        source_top1s.append(citizen_label)
    if "text" in present_weights and text_collapsed is not None:
        source_top1s.append(get_top1_with_tiebreak(text_collapsed))
    if "image" in present_weights and image_collapsed is not None:
        source_top1s.append(get_top1_with_tiebreak(image_collapsed))

    num_sources = len(source_top1s)
    if num_sources == 1:
        agreement = "SINGLE_SOURCE"
    elif num_sources == 2:
        agreement = "AGREE" if source_top1s[0] == source_top1s[1] else "CONFLICT"
    elif num_sources == 3:
        if source_top1s[0] == source_top1s[1] == source_top1s[2]:
            agreement = "AGREE"
        elif (
            source_top1s[0] == source_top1s[1]
            or source_top1s[0] == source_top1s[2]
            or source_top1s[1] == source_top1s[2]
        ):
            agreement = "MAJORITY"
        else:
            agreement = "CONFLICT"
    else:
        agreement = "SINGLE_SOURCE"

    # Label source (Section 7.6)
    if has_ml and "citizen" in present_weights:
        label_source = "FUSED"
    elif has_ml:
        label_source = "ML_ONLY"
    else:
        label_source = "USER_ONLY"

    return FusionResult(
        fused_probabilities=p_vector,
        top1=top1,
        final_label=final_label,
        model_confidence=model_confidence,
        confidence_level=confidence_level,
        margin=margin,
        agreement=agreement,
        label_source=label_source,
    )
