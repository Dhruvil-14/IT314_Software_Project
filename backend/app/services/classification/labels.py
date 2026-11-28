"""Labels, internal vectors, and model-to-scenario mappings (frozen per plan v2)."""

from typing import Final

PIPELINE_VERSION: Final[str] = "fusion-v2"

# Internal vector order used across all fusion operations (Section 5)
CLASSES: Final[tuple[str, ...]] = ("FLOOD", "FIRE", "CYCLONE", "OTHER")

# Supported public labels (Section 5)
FINAL_LABELS: Final[tuple[str, ...]] = ("FLOOD", "FIRE", "CYCLONE", "UNDETERMINED")

# Citizen scenario mapping (Section 5)
CITIZEN_SCENARIO_MAP: Final[dict[str, str]] = {
    "URBAN_FLOODING": "FLOOD",
    "INDUSTRIAL_FIRE": "FIRE",
    "CYCLONE": "CYCLONE",
}

# Scenario mapping back to platform database enum (Section 5)
LABEL_TO_SCENARIO_MAP: Final[dict[str, str]] = {
    "FLOOD": "URBAN_FLOODING",
    "FIRE": "INDUSTRIAL_FIRE",
    "CYCLONE": "CYCLONE",
}

# Scenario mapping back to lowercase application literal (Section 5, 10)
LABEL_TO_LOWER_SCENARIO_MAP: Final[dict[str, str]] = {
    "FLOOD": "urban_flooding",
    "FIRE": "industrial_fire",
    "CYCLONE": "cyclone",
}

# IndicBERT 5 classes mapping to 4 internal classes (Section 5)
TEXT_LABEL_MAP: Final[dict[str, str]] = {
    "flood": "FLOOD",
    "fire": "FIRE",
    "cyclone": "CYCLONE",
    "earthquake": "OTHER",
    "non-disaster": "OTHER",
    "non_disaster": "OTHER",
}

# ConvNeXt 7 classes mapping to 4 internal classes (Section 5)
IMAGE_LABEL_MAP: Final[dict[str, str]] = {
    "flood": "FLOOD",
    "fire": "FIRE",
    "hurricane": "CYCLONE",
    "earthquake": "OTHER",
    "landslide": "OTHER",
    "other_disaster": "OTHER",
    "not_disaster": "OTHER",
}


def map_citizen_scenario(scenario: str | None) -> str | None:
    """Map a citizen-submitted scenario (e.g. 'URBAN_FLOODING' or 'urban_flooding') to internal label."""
    if not scenario:
        return None
    normalized = scenario.strip().upper()
    return CITIZEN_SCENARIO_MAP.get(normalized)


def collapse_text_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """Collapse IndicBERT 5-class probabilities into 4 internal classes (FLOOD, FIRE, CYCLONE, OTHER)."""
    collapsed: dict[str, float] = {c: 0.0 for c in CLASSES}
    for label, prob in probabilities.items():
        norm_label = label.strip().lower()
        target = TEXT_LABEL_MAP.get(norm_label)
        if target and target in collapsed:
            collapsed[target] += float(prob)
        else:
            # Any unrecognised class maps into OTHER per Section 5
            collapsed["OTHER"] += float(prob)
    return collapsed


def collapse_image_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """Collapse ConvNeXt 7-class probabilities into 4 internal classes (FLOOD, FIRE, CYCLONE, OTHER)."""
    collapsed: dict[str, float] = {c: 0.0 for c in CLASSES}
    for label, prob in probabilities.items():
        norm_label = label.strip().lower()
        target = IMAGE_LABEL_MAP.get(norm_label)
        if target and target in collapsed:
            collapsed[target] += float(prob)
        else:
            collapsed["OTHER"] += float(prob)
    return collapsed


def map_final_label_to_scenario(final_label: str) -> str | None:
    """Map final label to DB scenario_type enum ('URBAN_FLOODING', 'INDUSTRIAL_FIRE', 'CYCLONE', or None)."""
    return LABEL_TO_SCENARIO_MAP.get(final_label)


def map_final_label_to_lower_scenario(final_label: str) -> str | None:
    """Map final label to application lowercase scenario ('urban_flooding', 'industrial_fire', 'cyclone', or None)."""
    return LABEL_TO_LOWER_SCENARIO_MAP.get(final_label)
