"""Report classification pipeline (fusion-v2)."""

from app.services.classification.fusion import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    WEIGHT_CITIZEN,
    WEIGHT_IMAGE,
    WEIGHT_TEXT,
    FusionResult,
    aggregate_photo_vectors,
    build_citizen_vector,
    fuse_sources,
)
from app.services.classification.labels import (
    CLASSES,
    FINAL_LABELS,
    PIPELINE_VERSION,
    collapse_image_probabilities,
    collapse_text_probabilities,
    map_citizen_scenario,
    map_final_label_to_lower_scenario,
    map_final_label_to_scenario,
)
from app.services.classification.review import (
    CONFLICT_MIN_PROBABILITY,
    REVIEW_LOW_CONFIDENCE_THRESHOLD,
    REVIEW_LOW_MARGIN_THRESHOLD,
    REVIEW_REASONS,
    evaluate_review_rules,
)

__all__ = [
    "CLASSES",
    "CONFIDENCE_HIGH_THRESHOLD",
    "CONFIDENCE_MEDIUM_THRESHOLD",
    "CONFLICT_MIN_PROBABILITY",
    "FINAL_LABELS",
    "PIPELINE_VERSION",
    "REVIEW_LOW_CONFIDENCE_THRESHOLD",
    "REVIEW_LOW_MARGIN_THRESHOLD",
    "REVIEW_REASONS",
    "WEIGHT_CITIZEN",
    "WEIGHT_IMAGE",
    "WEIGHT_TEXT",
    "FusionResult",
    "aggregate_photo_vectors",
    "build_citizen_vector",
    "collapse_image_probabilities",
    "collapse_text_probabilities",
    "evaluate_review_rules",
    "fuse_sources",
    "map_citizen_scenario",
    "map_final_label_to_lower_scenario",
    "map_final_label_to_scenario",
]
