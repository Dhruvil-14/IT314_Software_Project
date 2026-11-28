"""Unit tests for report classification pipeline (P1 Foundations).

Pins frozen constants from sections 5, 7, 8 and verifies collapse mappings,
worked examples, photo aggregation, tie-breaking, renormalisation, and review rules.
"""

import pytest

from app.services.classification.fusion import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    WEIGHT_CITIZEN,
    WEIGHT_IMAGE,
    WEIGHT_TEXT,
    aggregate_photo_vectors,
    build_citizen_vector,
    fuse_sources,
)
from app.services.classification.labels import (
    CITIZEN_SCENARIO_MAP,
    CLASSES,
    FINAL_LABELS,
    IMAGE_LABEL_MAP,
    LABEL_TO_LOWER_SCENARIO_MAP,
    LABEL_TO_SCENARIO_MAP,
    PIPELINE_VERSION,
    TEXT_LABEL_MAP,
    collapse_image_probabilities,
    collapse_text_probabilities,
    map_citizen_scenario,
    map_final_label_to_lower_scenario,
    map_final_label_to_scenario,
)
from app.services.classification.review import (
    CONFLICT_MIN_PROBABILITY,
    OTHER_DISASTER_CLASSES_IMAGE,
    OTHER_DISASTER_CLASSES_TEXT,
    REVIEW_LOW_CONFIDENCE_THRESHOLD,
    REVIEW_LOW_MARGIN_THRESHOLD,
    REVIEW_REASONS,
    evaluate_review_rules,
)

# ═══════════════════════════════════════════════════════════════════════
# 1. Constants Pinning Tests (Frozen Sections 5, 7, 8)
# ═══════════════════════════════════════════════════════════════════════

def test_constants_pinning_section_5():
    """Verify section 5 labels, order, and mappings are exactly as frozen."""
    assert PIPELINE_VERSION == "fusion-v2"
    assert CLASSES == ("FLOOD", "FIRE", "CYCLONE", "OTHER")
    assert FINAL_LABELS == ("FLOOD", "FIRE", "CYCLONE", "UNDETERMINED")

    assert CITIZEN_SCENARIO_MAP == {
        "URBAN_FLOODING": "FLOOD",
        "INDUSTRIAL_FIRE": "FIRE",
        "CYCLONE": "CYCLONE",
    }
    assert LABEL_TO_SCENARIO_MAP == {
        "FLOOD": "URBAN_FLOODING",
        "FIRE": "INDUSTRIAL_FIRE",
        "CYCLONE": "CYCLONE",
    }
    assert LABEL_TO_LOWER_SCENARIO_MAP == {
        "FLOOD": "urban_flooding",
        "FIRE": "industrial_fire",
        "CYCLONE": "cyclone",
    }

    # Text mapping (IndicBERT)
    assert TEXT_LABEL_MAP["flood"] == "FLOOD"
    assert TEXT_LABEL_MAP["fire"] == "FIRE"
    assert TEXT_LABEL_MAP["cyclone"] == "CYCLONE"
    assert TEXT_LABEL_MAP["earthquake"] == "OTHER"
    assert TEXT_LABEL_MAP["non-disaster"] == "OTHER"

    # Image mapping (ConvNeXt)
    assert IMAGE_LABEL_MAP["flood"] == "FLOOD"
    assert IMAGE_LABEL_MAP["fire"] == "FIRE"
    assert IMAGE_LABEL_MAP["hurricane"] == "CYCLONE"
    assert IMAGE_LABEL_MAP["earthquake"] == "OTHER"
    assert IMAGE_LABEL_MAP["landslide"] == "OTHER"
    assert IMAGE_LABEL_MAP["other_disaster"] == "OTHER"
    assert IMAGE_LABEL_MAP["not_disaster"] == "OTHER"


def test_constants_pinning_section_7():
    """Verify section 7 weights, citizen vector, and confidence thresholds."""
    assert WEIGHT_IMAGE == 0.45
    assert WEIGHT_TEXT == 0.35
    assert WEIGHT_CITIZEN == 0.20
    assert pytest.approx(WEIGHT_IMAGE + WEIGHT_TEXT + WEIGHT_CITIZEN, abs=1e-6) == 1.00

    u_vec = build_citizen_vector("FLOOD")
    assert u_vec["FLOOD"] == 0.70
    assert u_vec["FIRE"] == 0.10
    assert u_vec["CYCLONE"] == 0.10
    assert u_vec["OTHER"] == 0.10
    assert pytest.approx(sum(u_vec.values()), abs=1e-6) == 1.00

    assert CONFIDENCE_HIGH_THRESHOLD == 0.80
    assert CONFIDENCE_MEDIUM_THRESHOLD == 0.60


def test_constants_pinning_section_8():
    """Verify section 8 review rules, thresholds, and codes."""
    assert REVIEW_LOW_CONFIDENCE_THRESHOLD == 0.70
    assert REVIEW_LOW_MARGIN_THRESHOLD == 0.20
    assert CONFLICT_MIN_PROBABILITY == 0.60

    assert OTHER_DISASTER_CLASSES_IMAGE == ("earthquake", "landslide", "other_disaster")
    assert OTHER_DISASTER_CLASSES_TEXT == ("earthquake",)

    assert REVIEW_REASONS == (
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


# ═══════════════════════════════════════════════════════════════════════
# 2. Collapse Mapping Tests
# ═══════════════════════════════════════════════════════════════════════

def test_collapse_text_probabilities():
    raw_text = {
        "Flood": 0.82,
        "Cyclone": 0.08,
        "Fire": 0.03,
        "Earthquake": 0.02,
        "Non-Disaster": 0.05,
    }
    t = collapse_text_probabilities(raw_text)
    assert pytest.approx(t["FLOOD"]) == 0.82
    assert pytest.approx(t["FIRE"]) == 0.03
    assert pytest.approx(t["CYCLONE"]) == 0.08
    assert pytest.approx(t["OTHER"]) == 0.07
    assert pytest.approx(sum(t.values())) == 1.00


def test_collapse_image_probabilities():
    raw_image = {
        "flood": 0.91,
        "hurricane": 0.04,
        "fire": 0.01,
        "earthquake": 0.01,
        "landslide": 0.01,
        "other_disaster": 0.01,
        "not_disaster": 0.01,
    }
    img = collapse_image_probabilities(raw_image)
    assert pytest.approx(img["FLOOD"]) == 0.91
    assert pytest.approx(img["FIRE"]) == 0.01
    assert pytest.approx(img["CYCLONE"]) == 0.04
    assert pytest.approx(img["OTHER"]) == 0.04
    assert pytest.approx(sum(img.values())) == 1.00


def test_scenario_helpers():
    assert map_citizen_scenario("urban_flooding") == "FLOOD"
    assert map_citizen_scenario("INDUSTRIAL_FIRE") == "FIRE"
    assert map_citizen_scenario("CYCLONE") == "CYCLONE"
    assert map_citizen_scenario("EARTHQUAKE") is None
    assert map_citizen_scenario(None) is None

    assert map_final_label_to_scenario("FLOOD") == "URBAN_FLOODING"
    assert map_final_label_to_scenario("FIRE") == "INDUSTRIAL_FIRE"
    assert map_final_label_to_scenario("CYCLONE") == "CYCLONE"
    assert map_final_label_to_scenario("UNDETERMINED") is None

    assert map_final_label_to_lower_scenario("FLOOD") == "urban_flooding"
    assert map_final_label_to_lower_scenario("FIRE") == "industrial_fire"
    assert map_final_label_to_lower_scenario("CYCLONE") == "cyclone"
    assert map_final_label_to_lower_scenario("UNDETERMINED") is None


# ═══════════════════════════════════════════════════════════════════════
# 3. Photo Aggregation Tests (Section 7.1)
# ═══════════════════════════════════════════════════════════════════════

def test_aggregate_photos_filters_not_disaster_when_relevant_photo_exists():
    photos = [
        {
            "prediction": "flood",
            "all_probabilities": {"flood": 0.90, "fire": 0.05, "hurricane": 0.03, "not_disaster": 0.02},
            "needs_review": False,
        },
        {
            "prediction": "not_disaster",
            "all_probabilities": {"not_disaster": 0.95, "flood": 0.05},
            "needs_review": False,
        },
    ]
    mean_vec, kept = aggregate_photo_vectors(photos)
    assert len(kept) == 1
    assert kept[0]["prediction"] == "flood"
    assert pytest.approx(mean_vec["FLOOD"]) == 0.90


def test_aggregate_photos_keeps_all_when_all_are_not_disaster():
    photos = [
        {
            "prediction": "not_disaster",
            "all_probabilities": {"not_disaster": 0.90, "flood": 0.10},
            "needs_review": False,
        },
        {
            "prediction": "not_disaster",
            "all_probabilities": {"not_disaster": 0.80, "flood": 0.20},
            "needs_review": False,
        },
    ]
    mean_vec, kept = aggregate_photo_vectors(photos)
    assert len(kept) == 2
    assert pytest.approx(mean_vec["FLOOD"]) == 0.15
    assert pytest.approx(mean_vec["OTHER"]) == 0.85


def test_aggregate_photos_empty_list():
    mean_vec, kept = aggregate_photo_vectors([])
    assert mean_vec is None
    assert kept == []


# ═══════════════════════════════════════════════════════════════════════
# 4. Worked Examples from Section 7.7
# ═══════════════════════════════════════════════════════════════════════

def test_worked_example_a_everything_agrees():
    """Hand-checked example A from Section 7.7:
    Citizen: flooding (URBAN_FLOODING)
    Text: Flood 0.82, Cyclone 0.08, Fire 0.03, Earthquake 0.02, Non-Disaster 0.05
    Image: flood 0.91, hurricane 0.04, fire 0.01, rest 0.04
    Expected: FLOOD, conf 0.8365 (0.837), HIGH, margin 0.771, AGREE, no review.
    """
    cit_label = "FLOOD"
    text_collapsed = {"FLOOD": 0.82, "FIRE": 0.03, "CYCLONE": 0.08, "OTHER": 0.07}
    img_collapsed = {"FLOOD": 0.91, "FIRE": 0.01, "CYCLONE": 0.04, "OTHER": 0.04}

    res = fuse_sources(cit_label, text_collapsed, img_collapsed)
    assert res.final_label == "FLOOD"
    assert res.top1 == "FLOOD"
    assert pytest.approx(res.fused_probabilities["FLOOD"], abs=1e-4) == 0.8365
    assert pytest.approx(res.fused_probabilities["FIRE"], abs=1e-4) == 0.0350
    assert pytest.approx(res.fused_probabilities["CYCLONE"], abs=1e-4) == 0.0660
    assert pytest.approx(res.fused_probabilities["OTHER"], abs=1e-4) == 0.0625
    assert pytest.approx(res.model_confidence, abs=1e-3) == 0.837
    assert res.confidence_level == "HIGH"
    assert pytest.approx(res.margin, abs=1e-3) == 0.771
    assert res.agreement == "AGREE"
    assert res.label_source == "FUSED"

    needs_review, reasons = evaluate_review_rules(
        final_label=res.final_label,
        top1=res.top1,
        model_confidence=res.model_confidence,
        margin=res.margin,
        user_label=cit_label,
        text_top1_raw="Flood",
        text_top1_collapsed="FLOOD",
        text_top1_prob=0.82,
        image_top1_raw="flood",
        image_top1_collapsed="FLOOD",
        image_top1_prob=0.91,
        any_image_needs_review=False,
        ml_models_available=True,
    )
    assert not needs_review
    assert reasons == []


def test_worked_example_b_earthquake_and_fire():
    """Hand-checked example B from Section 7.7:
    Text says earthquake (0.70), citizen says fire, no photo.
    Expected: UNDETERMINED, confidence 0.545 (LOW), margin 0.227.
    Review reasons: NO_SUPPORTED_LABEL, PREDICTED_OTHER_DISASTER, LOW_CONFIDENCE, USER_LABEL_DISAGREES.
    """
    cit_label = "FIRE"
    text_collapsed = {"FLOOD": 0.05, "FIRE": 0.10, "CYCLONE": 0.05, "OTHER": 0.80}
    img_collapsed = None

    res = fuse_sources(cit_label, text_collapsed, img_collapsed)
    assert res.final_label == "UNDETERMINED"
    assert res.top1 == "OTHER"
    assert pytest.approx(res.fused_probabilities["OTHER"], abs=1e-3) == 0.545
    assert pytest.approx(res.fused_probabilities["FIRE"], abs=1e-3) == 0.318
    assert pytest.approx(res.fused_probabilities["FLOOD"], abs=1e-3) == 0.068
    assert pytest.approx(res.fused_probabilities["CYCLONE"], abs=1e-3) == 0.068
    assert pytest.approx(res.model_confidence, abs=1e-3) == 0.545
    assert res.confidence_level == "LOW"
    assert pytest.approx(res.margin, abs=1e-3) == 0.227
    assert res.agreement == "CONFLICT"
    assert res.label_source == "FUSED"

    needs_review, reasons = evaluate_review_rules(
        final_label=res.final_label,
        top1=res.top1,
        model_confidence=res.model_confidence,
        margin=res.margin,
        user_label=cit_label,
        text_top1_raw="Earthquake",
        text_top1_collapsed="OTHER",
        text_top1_prob=0.70,
        image_top1_raw=None,
        image_top1_collapsed=None,
        image_top1_prob=None,
        any_image_needs_review=False,
        ml_models_available=True,
    )
    assert needs_review
    assert sorted(reasons) == sorted([
        "NO_SUPPORTED_LABEL",
        "PREDICTED_OTHER_DISASTER",
        "LOW_CONFIDENCE",
        "USER_LABEL_DISAGREES",
    ])


def test_worked_example_c_both_services_down():
    """Hand-checked example C from Section 7.7:
    Both ML services down. Citizen picked cyclone.
    Expected: CYCLONE, label_source = USER_ONLY, model_confidence = null,
    confidence_level = null, review reason MODELS_UNAVAILABLE.
    """
    cit_label = "CYCLONE"
    res = fuse_sources(cit_label, text_collapsed=None, image_collapsed=None)
    assert res.final_label == "CYCLONE"
    assert res.top1 == "CYCLONE"
    assert res.model_confidence is None
    assert res.confidence_level is None
    assert res.label_source == "USER_ONLY"
    assert res.agreement == "SINGLE_SOURCE"

    needs_review, reasons = evaluate_review_rules(
        final_label=res.final_label,
        top1=res.top1,
        model_confidence=res.model_confidence,
        margin=res.margin,
        user_label=cit_label,
        text_top1_raw=None,
        text_top1_collapsed=None,
        text_top1_prob=None,
        image_top1_raw=None,
        image_top1_collapsed=None,
        image_top1_prob=None,
        any_image_needs_review=False,
        ml_models_available=False,
    )
    assert needs_review
    assert reasons == ["MODELS_UNAVAILABLE"]


# ═══════════════════════════════════════════════════════════════════════
# 5. Tie-breaking and Renormalisation Tests
# ═══════════════════════════════════════════════════════════════════════

def test_tie_breaking_order():
    """Verify tie-breaking order is strictly FLOOD, FIRE, CYCLONE, OTHER (Section 7.3)."""
    # 1. FLOOD vs FIRE
    p1 = {"FLOOD": 0.40, "FIRE": 0.40, "CYCLONE": 0.10, "OTHER": 0.10}
    from app.services.classification.fusion import get_top1_with_tiebreak
    assert get_top1_with_tiebreak(p1) == "FLOOD"

    # 2. FIRE vs CYCLONE
    p2 = {"FLOOD": 0.10, "FIRE": 0.40, "CYCLONE": 0.40, "OTHER": 0.10}
    assert get_top1_with_tiebreak(p2) == "FIRE"

    # 3. CYCLONE vs OTHER
    p3 = {"FLOOD": 0.10, "FIRE": 0.10, "CYCLONE": 0.40, "OTHER": 0.40}
    assert get_top1_with_tiebreak(p3) == "CYCLONE"

    # 4. All four equal (0.25 each)
    p4 = {"FLOOD": 0.25, "FIRE": 0.25, "CYCLONE": 0.25, "OTHER": 0.25}
    assert get_top1_with_tiebreak(p4) == "FLOOD"


def test_renormalisation_drops_missing_sources():
    # Only image present (w=0.45) -> renormalised to 1.0
    img = {"FLOOD": 0.70, "FIRE": 0.10, "CYCLONE": 0.10, "OTHER": 0.10}
    res = fuse_sources(citizen_label=None, text_collapsed=None, image_collapsed=img)
    assert res.label_source == "ML_ONLY"
    assert pytest.approx(res.fused_probabilities["FLOOD"]) == 0.70
    assert pytest.approx(sum(res.fused_probabilities.values())) == 1.00


# ═══════════════════════════════════════════════════════════════════════
# 6. Every Review Rule On / Off Tests (Section 8)
# ═══════════════════════════════════════════════════════════════════════

def test_review_rule_predicted_not_disaster():
    """When OTHER won, but neither image nor text was earthquake/landslide/other_disaster."""
    needs_review, reasons = evaluate_review_rules(
        final_label="UNDETERMINED",
        top1="OTHER",
        model_confidence=0.85,
        margin=0.40,
        user_label=None,
        text_top1_raw="Non-Disaster",
        text_top1_collapsed="OTHER",
        text_top1_prob=0.85,
        image_top1_raw="not_disaster",
        image_top1_collapsed="OTHER",
        image_top1_prob=0.85,
        any_image_needs_review=False,
        ml_models_available=True,
    )
    assert needs_review
    assert "NO_SUPPORTED_LABEL" in reasons
    assert "PREDICTED_NOT_DISASTER" in reasons
    assert "PREDICTED_OTHER_DISASTER" not in reasons


def test_review_rule_low_confidence():
    needs_review, reasons = evaluate_review_rules(
        final_label="FLOOD",
        top1="FLOOD",
        model_confidence=0.69,  # < 0.70
        margin=0.30,
        user_label="FLOOD",
        text_top1_raw="Flood",
        text_top1_collapsed="FLOOD",
        text_top1_prob=0.69,
        image_top1_raw=None,
        image_top1_collapsed=None,
        image_top1_prob=None,
        any_image_needs_review=False,
        ml_models_available=True,
    )
    assert needs_review
    assert "LOW_CONFIDENCE" in reasons


def test_review_rule_low_margin():
    needs_review, reasons = evaluate_review_rules(
        final_label="FLOOD",
        top1="FLOOD",
        model_confidence=0.75,
        margin=0.19,  # < 0.20
        user_label="FLOOD",
        text_top1_raw="Flood",
        text_top1_collapsed="FLOOD",
        text_top1_prob=0.75,
        image_top1_raw=None,
        image_top1_collapsed=None,
        image_top1_prob=None,
        any_image_needs_review=False,
        ml_models_available=True,
    )
    assert needs_review
    assert "LOW_MARGIN" in reasons


def test_review_rule_user_label_disagrees():
    needs_review, reasons = evaluate_review_rules(
        final_label="FIRE",
        top1="FIRE",
        model_confidence=0.85,
        margin=0.50,
        user_label="FLOOD",  # disagrees with final_label
        text_top1_raw="Fire",
        text_top1_collapsed="FIRE",
        text_top1_prob=0.85,
        image_top1_raw=None,
        image_top1_collapsed=None,
        image_top1_prob=None,
        any_image_needs_review=False,
        ml_models_available=True,
    )
    assert needs_review
    assert "USER_LABEL_DISAGREES" in reasons


def test_review_rule_text_image_conflict():
    # Both >= 0.60, but collapsed predictions differ
    needs_review, reasons = evaluate_review_rules(
        final_label="FLOOD",
        top1="FLOOD",
        model_confidence=0.72,
        margin=0.25,
        user_label="FLOOD",
        text_top1_raw="Fire",
        text_top1_collapsed="FIRE",
        text_top1_prob=0.65,
        image_top1_raw="flood",
        image_top1_collapsed="FLOOD",
        image_top1_prob=0.70,
        any_image_needs_review=False,
        ml_models_available=True,
    )
    assert needs_review
    assert "TEXT_IMAGE_CONFLICT" in reasons


def test_review_rule_image_needs_review():
    needs_review, reasons = evaluate_review_rules(
        final_label="FLOOD",
        top1="FLOOD",
        model_confidence=0.85,
        margin=0.40,
        user_label="FLOOD",
        text_top1_raw="Flood",
        text_top1_collapsed="FLOOD",
        text_top1_prob=0.85,
        image_top1_raw="flood",
        image_top1_collapsed="FLOOD",
        image_top1_prob=0.85,
        any_image_needs_review=True,  # triggers rule
        ml_models_available=True,
    )
    assert needs_review
    assert "IMAGE_NEEDS_REVIEW" in reasons
