"""Unit and worker orchestration tests for classification queue, retries, and sweep (Phase 2)."""

import uuid
from unittest.mock import MagicMock, patch

from app.services.classification.clients import TextPredictionResult
from app.services.classification.worker import (
    MAX_ATTEMPTS,
    RETRY_DELAYS_SECONDS,
    SWEEP_TIMEOUT_MINUTES,
    compute_input_hash,
    queue_classification,
    run_classification,
    sweep_stale_classifications,
)


def test_worker_constants():
    """Verify Section 14 retry schedule and sweep timeout constants are frozen."""
    assert RETRY_DELAYS_SECONDS == (30, 120, 600)
    assert MAX_ATTEMPTS == 3
    assert SWEEP_TIMEOUT_MINUTES == 10


def test_compute_input_hash_is_deterministic_and_sorts_photos():
    h1 = compute_input_hash("Flooding in street", ["hash_b", "hash_a"], "FLOOD")
    h2 = compute_input_hash("Flooding in street", ["hash_a", "hash_b"], "FLOOD")
    assert h1 == h2

    # Varying description changes hash
    h3 = compute_input_hash("Different description", ["hash_a", "hash_b"], "FLOOD")
    assert h1 != h3

    # Varying user label changes hash
    h4 = compute_input_hash("Flooding in street", ["hash_a", "hash_b"], "FIRE")
    assert h1 != h4


def test_queue_classification_skips_when_mode_is_off():
    mock_conn = MagicMock()
    with patch("app.services.classification.worker.get_settings") as mock_settings:
        mock_settings.return_value.classifier_mode = "off"
        qid = queue_classification(mock_conn, uuid.uuid4(), "SUBMITTED")
        assert qid is None
        assert not mock_conn.execute.called


def test_run_classification_skips_when_mode_is_off():
    with patch("app.services.classification.worker.get_settings") as mock_settings:
        mock_settings.return_value.classifier_mode = "off"
        res = run_classification(uuid.uuid4())
        assert res is None


def test_queue_classification_deduplicates_existing_queued_hash():
    mock_conn = MagicMock()
    inc_id = uuid.uuid4()
    existing_qid = uuid.uuid4()

    with patch("app.services.classification.worker.get_settings") as mock_settings:
        mock_settings.return_value.classifier_mode = "live"

        # Mock incident lookup
        mock_conn.execute.return_value.fetchone.side_effect = [
            {"scenario": "URBAN_FLOODING", "description": "Flooding near bridge"},  # incidents query
            {"classification_id": existing_qid},  # existing check query
        ]
        mock_conn.execute.return_value.fetchall.return_value = []  # photos query

        qid = queue_classification(mock_conn, inc_id, "SUBMITTED")
        assert qid == existing_qid


def test_sweep_stale_classifications_resets_running_rows():
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [
        {"classification_id": uuid.uuid4()},
        {"classification_id": uuid.uuid4()},
    ]

    with patch("app.db.transaction") as mock_tx:
        mock_tx.return_value.__enter__.return_value = mock_conn
        reset_count = sweep_stale_classifications(timeout_minutes=10)
        assert reset_count == 2
        assert mock_conn.execute.called
        query_sql = mock_conn.execute.call_args[0][0]
        assert "WHERE status = 'RUNNING'" in query_sql


def test_run_classification_completes_when_ml_succeeds():
    inc_id = uuid.uuid4()
    job_id = uuid.uuid4()

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.side_effect = [
        # 1. Claim job query
        {
            "classification_id": job_id,
            "incident_id": inc_id,
            "attempts": 0,
            "user_label": "FLOOD",
        },
        # 2. _load_incident_and_photos query
        {
            "incident_id": inc_id,
            "incident_ref": "INC-2026-0001",
            "scenario": "URBAN_FLOODING",
            "description": "Rising waters in homes",
        },
    ]
    mock_conn.execute.return_value.fetchall.return_value = []

    text_mock = TextPredictionResult(
        raw_prediction="Flood",
        confidence=0.85,
        raw_probabilities={"Flood": 0.85, "Fire": 0.05, "Cyclone": 0.05, "Earthquake": 0.02, "Non-Disaster": 0.03},
        collapsed_probabilities={"FLOOD": 0.85, "FIRE": 0.05, "CYCLONE": 0.05, "OTHER": 0.05},
        collapsed_top1="FLOOD",
        collapsed_top1_prob=0.85,
    )

    with (
        patch("app.db.transaction") as mock_tx,
        patch("app.audit.record") as mock_audit,
        patch("app.services.classification.worker.get_settings") as mock_settings,
        patch("app.services.classification.worker.classify_text", return_value=text_mock),
    ):
        mock_tx.return_value.__enter__.return_value = mock_conn
        mock_settings.return_value.classifier_mode = "live"
        mock_settings.return_value.disaster_classifier_url = "https://text-classifier.test/predict"
        mock_settings.return_value.ai_image_service_url = None

        res = run_classification(job_id)

        assert res is not None
        assert res["status"] == "COMPLETED"
        assert res["final_label"] == "FLOOD"
        assert not res["human_review_required"]
        assert mock_audit.called
        assert mock_audit.call_args[1]["action"] == "CLASSIFICATION_COMPLETED"


def test_run_classification_degrades_when_models_fail_after_max_attempts():
    inc_id = uuid.uuid4()
    job_id = uuid.uuid4()

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.side_effect = [
        # 1. Claim job query (already on attempt 3)
        {
            "classification_id": job_id,
            "incident_id": inc_id,
            "attempts": 3,
            "user_label": "CYCLONE",
        },
        # 2. _load_incident_and_photos query
        {
            "incident_id": inc_id,
            "incident_ref": "INC-2026-0002",
            "scenario": "CYCLONE",
            "description": "High winds blowing trees down",
        },
    ]
    mock_conn.execute.return_value.fetchall.return_value = []

    with (
        patch("app.db.transaction") as mock_tx,
        patch("app.audit.record") as mock_audit,
        patch("app.services.classification.worker.get_settings") as mock_settings,
        patch("app.services.classification.worker.classify_text", return_value=None),
    ):
        mock_tx.return_value.__enter__.return_value = mock_conn
        mock_settings.return_value.classifier_mode = "live"
        mock_settings.return_value.disaster_classifier_url = "https://text-classifier.test/predict"
        mock_settings.return_value.ai_image_service_url = None

        res = run_classification(job_id)

        assert res is not None
        assert res["status"] == "DEGRADED"
        assert res["final_label"] == "CYCLONE"
        assert res["label_source"] == "USER_ONLY"
        assert res["human_review_required"]
        assert "MODELS_UNAVAILABLE" in res["review_reasons"]
        assert mock_audit.called
        assert mock_audit.call_args[1]["action"] == "CLASSIFICATION_DEGRADED"
