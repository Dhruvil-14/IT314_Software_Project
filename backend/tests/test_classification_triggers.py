"""Tests for classification triggers: citizen endpoints, sweep API, and CLI (Phase 3)."""

import argparse
import uuid
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.routers.citizen import _store_attachment
from main import app

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════════
# 1. Internal Classification Sweep Endpoint Tests
# ═══════════════════════════════════════════════════════════════════════

def test_sweep_endpoint_rejects_missing_token():
    resp = client.post("/api/v1/internal/classification/sweep")
    assert resp.status_code == 401


def test_sweep_endpoint_rejects_invalid_token():
    with patch("app.routers.internal.get_settings") as mock_settings:
        mock_settings.return_value.classifier_sweep_token = "valid-token-12345"
        resp = client.post(
            "/api/v1/internal/classification/sweep",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401


def test_sweep_endpoint_accepts_valid_token_header():
    with (
        patch("app.routers.internal.get_settings") as mock_settings,
        patch("app.services.classification.worker.sweep_stale_classifications", return_value=3),
        patch("app.services.classification.worker.run_classification", side_effect=[{"status": "COMPLETED"}, None]),
    ):
        mock_settings.return_value.classifier_sweep_token = "valid-token-12345"
        resp = client.post(
            "/api/v1/internal/classification/sweep",
            headers={"X-Classification-Sweep-Token": "valid-token-12345"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["reset_count"] == 3
        assert data["processed_count"] == 1
        assert "Swept 3 stale running jobs" in data["message"]


def test_sweep_endpoint_accepts_bearer_auth():
    with (
        patch("app.routers.internal.get_settings") as mock_settings,
        patch("app.services.classification.worker.sweep_stale_classifications", return_value=0),
        patch("app.services.classification.worker.run_classification", return_value=None),
    ):
        mock_settings.return_value.classifier_sweep_token = "valid-token-12345"
        resp = client.post(
            "/api/v1/internal/classification/sweep",
            headers={"Authorization": "Bearer valid-token-12345"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["reset_count"] == 0
        assert data["processed_count"] == 0


# ═══════════════════════════════════════════════════════════════════════
# 2. CLI Command Tests
# ═══════════════════════════════════════════════════════════════════════

def test_cli_classify_pending(capsys):
    from app.cli import classify_pending

    args = argparse.Namespace(limit=5)
    with (
        patch("app.services.classification.worker.sweep_stale_classifications", return_value=1),
        patch("app.services.classification.worker.run_classification", side_effect=[
            {"classification_id": uuid.uuid4(), "status": "COMPLETED", "final_label": "FLOOD"},
            None,
        ]),
    ):
        classify_pending(args)
        out = capsys.readouterr().out
        assert "Swept 1 stale rows." in out
        assert "Processed classification" in out
        assert "Completed 1 pending classifications." in out


# ═══════════════════════════════════════════════════════════════════════
# 3. Citizen Router Trigger Hooks Tests
# ═══════════════════════════════════════════════════════════════════════

def test_store_attachment_triggers_photo_added_queue():
    inc_id = uuid.uuid4()
    principal = MagicMock()
    principal.user_id = uuid.uuid4()
    principal.ip = "127.0.0.1"

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.side_effect = [
        {"status": "SUBMITTED"},  # status query
        None,  # duplicate sha256 check
        {"n": 1},  # count check (< 5)
        {"evidence_id": uuid.uuid4(), "file_name": "scene.jpg", "mime_type": "image/jpeg", "size_bytes": 100, "scan_state": "PENDING", "uploaded_at": None},  # insert return
    ]

    with (
        patch("app.db.transaction") as mock_tx,
        patch("app.routers.citizen._own_incident_status"),
        patch("app.routers.citizen.queue_classification") as mock_queue,
        patch("app.audit.record"),
    ):
        mock_tx.return_value.__enter__.return_value = mock_conn
        _body, created = _store_attachment(principal, inc_id, "scene.jpg", "image/jpeg", b"fake_content")

        assert created is True
        assert mock_queue.called
        assert mock_queue.call_args[0][1] == inc_id
        assert mock_queue.call_args[1]["trigger_event"] == "PHOTO_ADDED"
