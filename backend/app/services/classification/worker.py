"""Background classification worker, queueing, retries, and sweep (plan v2)."""

import hashlib
import logging
import uuid
from typing import Any

from psycopg import Connection
from psycopg.types.json import Jsonb

from app import audit, db
from app.config import get_settings
from app.services.classification.clients import (
    ImagePredictionResult,
    TextPredictionResult,
    classify_image,
    classify_text,
)
from app.services.classification.fusion import (
    aggregate_photo_vectors,
    fuse_sources,
)
from app.services.classification.labels import (
    PIPELINE_VERSION,
    map_citizen_scenario,
    map_final_label_to_scenario,
)
from app.services.classification.review import evaluate_review_rules

logger = logging.getLogger("drch.classification.worker")

# Retry backoff intervals (Section 14: 30s, 2 min, 10 min)
RETRY_DELAYS_SECONDS = (30, 120, 600)
MAX_ATTEMPTS = len(RETRY_DELAYS_SECONDS)
SWEEP_TIMEOUT_MINUTES = 10


def compute_input_hash(
    description: str,
    photo_hashes: list[str],
    user_label: str | None,
    pipeline_version: str = PIPELINE_VERSION,
) -> str:
    """Compute sha256 input hash for deduplication per Section 9."""
    sorted_photos = ",".join(sorted(photo_hashes))
    raw = f"desc:{description.strip()}|photos:{sorted_photos}|user:{user_label or ''}|ver:{pipeline_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def queue_classification(
    conn: Connection,
    incident_id: uuid.UUID,
    trigger_event: str,
    user_label: str | None = None,
) -> uuid.UUID | None:
    """Queue a classification version for an incident.

    - Respects CLASSIFIER_MODE ('off' queues nothing).
    - Checks uq_classification_input to suppress duplicate work.
    - Sets previous is_current=False, increments version, and creates new row in one transaction.
    """
    settings = get_settings()
    mode = getattr(settings, "classifier_mode", "off").lower()
    if mode == "off":
        logger.debug("Classifier mode is off; skipping queueing for incident %s", incident_id)
        return None

    # Load incident info for input_hash
    inc_row = conn.execute(
        "SELECT scenario, description FROM incidents WHERE incident_id = %s FOR UPDATE",
        (incident_id,),
    ).fetchone()
    if not inc_row:
        logger.warning("Cannot queue classification: incident %s not found", incident_id)
        return None

    resolved_user_label = user_label or map_citizen_scenario(inc_row.get("scenario"))

    # Load attached active photo hashes
    photo_rows = conn.execute(
        "SELECT sha256 FROM evidence_files WHERE incident_id = %s AND scan_state != 'REJECTED'",
        (incident_id,),
    ).fetchall()
    photo_hashes = [str(r["sha256"]) for r in photo_rows if r.get("sha256")]

    input_hash = compute_input_hash(
        description=inc_row.get("description", ""),
        photo_hashes=photo_hashes,
        user_label=resolved_user_label,
        pipeline_version=PIPELINE_VERSION,
    )

    # Check for existing duplicate in QUEUED/RUNNING/COMPLETED state
    existing = conn.execute(
        """
        SELECT classification_id FROM incident_classifications
        WHERE incident_id = %s AND input_hash = %s AND pipeline_version = %s
          AND status IN ('QUEUED', 'RUNNING', 'COMPLETED')
        LIMIT 1
        """,
        (incident_id, input_hash, PIPELINE_VERSION),
    ).fetchone()
    if existing:
        logger.info(
            "Identical input hash %s already queued/processed for incident %s; suppressing duplicate",
            input_hash[:8],
            incident_id,
        )
        return existing["classification_id"]

    # Calculate next version
    ver_row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 AS next_ver FROM incident_classifications WHERE incident_id = %s",
        (incident_id,),
    ).fetchone()
    next_ver = ver_row["next_ver"] if ver_row else 1

    # Mark previous current rows as false
    conn.execute(
        "UPDATE incident_classifications SET is_current = FALSE WHERE incident_id = %s AND is_current = TRUE",
        (incident_id,),
    )

    # Insert new QUEUED row
    new_row = conn.execute(
        """
        INSERT INTO incident_classifications (
            incident_id, version, is_current, status, trigger_event,
            pipeline_version, input_hash, attempts, next_attempt_at, user_label
        ) VALUES (
            %s, %s, TRUE, 'QUEUED', %s,
            %s, %s, 0, now(), %s
        )
        RETURNING classification_id
        """,
        (
            incident_id,
            next_ver,
            trigger_event,
            PIPELINE_VERSION,
            input_hash,
            resolved_user_label,
        ),
    ).fetchone()

    logger.info(
        "Queued classification %s (v%d, trigger=%s) for incident %s",
        new_row["classification_id"],
        next_ver,
        trigger_event,
        incident_id,
    )
    return new_row["classification_id"]


def sweep_stale_classifications(timeout_minutes: int = SWEEP_TIMEOUT_MINUTES) -> int:
    """Reset jobs stuck in RUNNING for longer than timeout_minutes back to QUEUED."""
    with db.transaction() as conn:
        reset_rows = conn.execute(
            """
            UPDATE incident_classifications
            SET status = 'QUEUED', next_attempt_at = now()
            WHERE status = 'RUNNING'
              AND (now() - next_attempt_at) > (%s * interval '1 minute')
            RETURNING classification_id
            """,
            (timeout_minutes,),
        ).fetchall()
        count = len(reset_rows)
        if count > 0:
            logger.warning("Reset %d stale RUNNING classifications back to QUEUED", count)
        return count


def _load_incident_and_photos(conn: Connection, incident_id: uuid.UUID) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    inc = conn.execute(
        "SELECT incident_id, incident_ref, scenario, description FROM incidents WHERE incident_id = %s",
        (incident_id,),
    ).fetchone()
    if not inc:
        return None, []

    photos = conn.execute(
        """
        SELECT ef.evidence_id, ef.file_name, ef.mime_type, ef.sha256, eb.content
        FROM evidence_files ef
        JOIN evidence_blobs eb ON ef.evidence_id = eb.evidence_id
        WHERE ef.incident_id = %s AND ef.scan_state != 'REJECTED'
        ORDER BY ef.uploaded_at ASC
        """,
        (incident_id,),
    ).fetchall()
    return inc, photos


def run_classification(classification_id: uuid.UUID | None = None) -> dict[str, Any] | None:
    """Poll and execute one classification job.

    Performs locking with FOR UPDATE SKIP LOCKED, executes model calls,
    handles cold start retries up to 3 times, fuses signals, applies review rules,
    and updates the database with audit logging.
    """
    settings = get_settings()
    mode = getattr(settings, "classifier_mode", "off").lower()
    if mode == "off":
        return None

    # Step 1: Claim job
    with db.transaction() as conn:
        if classification_id:
            row = conn.execute(
                """
                SELECT * FROM incident_classifications
                WHERE classification_id = %s FOR UPDATE
                """,
                (classification_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM incident_classifications
                WHERE status = 'QUEUED' AND next_attempt_at <= now()
                ORDER BY next_attempt_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """
            ).fetchone()

        if not row:
            return None

        job_id = row["classification_id"]
        incident_id = row["incident_id"]
        attempts = row["attempts"]
        user_label = row["user_label"]

        # Mark RUNNING
        conn.execute(
            "UPDATE incident_classifications SET status = 'RUNNING' WHERE classification_id = %s",
            (job_id,),
        )

        inc_info, photos = _load_incident_and_photos(conn, incident_id)

    if not inc_info:
        logger.error("Incident %s vanished while processing classification %s", incident_id, job_id)
        with db.transaction() as conn:
            conn.execute(
                "UPDATE incident_classifications SET status = 'FAILED', error_code = 'INCIDENT_NOT_FOUND' WHERE classification_id = %s",
                (job_id,),
            )
        return None

    # Step 2: Call ML services outside database transaction
    desc = inc_info.get("description", "")
    cit_label = user_label or map_citizen_scenario(inc_info.get("scenario"))

    # Text model
    text_res: TextPredictionResult | None = None
    text_url = getattr(settings, "disaster_classifier_url", None)
    if text_url:
        text_res = classify_text(desc, endpoint_url=text_url)

    # Image model
    image_results: list[ImagePredictionResult] = []
    image_base_url = getattr(settings, "ai_image_service_url", None)
    if image_base_url and photos:
        for p in photos:
            img_res = classify_image(
                image_bytes=bytes(p["content"]),
                filename=p.get("file_name", "photo.jpg"),
                content_type=p.get("mime_type", "image/jpeg"),
                evidence_id=str(p["evidence_id"]),
                base_url=image_base_url,
            )
            if img_res is not None:
                image_results.append(img_res)

    # Assess ML success vs failure
    text_configured = bool(text_url)
    image_configured = bool(image_base_url and photos)
    text_failed = text_configured and (text_res is None)
    image_failed = image_configured and (len(image_results) == 0)

    # Step 3: Handle transient retries (Section 14)
    # If any configured ML source failed and we have retries remaining:
    has_transient_failure = text_failed or image_failed
    if has_transient_failure and attempts < MAX_ATTEMPTS:
        retry_delay = RETRY_DELAYS_SECONDS[attempts]
        with db.transaction() as conn:
            conn.execute(
                """
                UPDATE incident_classifications
                SET status = 'QUEUED',
                    attempts = attempts + 1,
                    next_attempt_at = now() + (%s * interval '1 second')
                WHERE classification_id = %s
                """,
                (retry_delay, job_id),
            )
        logger.warning(
            "Classification %s encountered transient ML failure; scheduled retry %d in %ds",
            job_id,
            attempts + 1,
            retry_delay,
        )
        return {"status": "RETRY_SCHEDULED", "attempt": attempts + 1, "retry_in_seconds": retry_delay}

    # Step 4: Proceed with available sources (COMPLETED or DEGRADED)
    text_collapsed = text_res.collapsed_probabilities if text_res else None
    image_collapsed: dict[str, float] | None = None
    if image_results:
        photo_payloads = [
            {
                "prediction": img.raw_prediction,
                "all_probabilities": img.all_probabilities,
                "needs_review": img.needs_review,
            }
            for img in image_results
        ]
        image_collapsed, _ = aggregate_photo_vectors(photo_payloads)

    ml_succeeded = (text_collapsed is not None) or (image_collapsed is not None)
    status = "COMPLETED" if ml_succeeded else "DEGRADED"

    # Step 5: Perform probability fusion (Section 7)
    fusion = fuse_sources(
        citizen_label=cit_label,
        text_collapsed=text_collapsed,
        image_collapsed=image_collapsed,
    )

    # Step 6: Evaluate review rules (Section 8)
    any_img_review = any(img.needs_review for img in image_results)
    text_top1_raw = text_res.raw_prediction if text_res else None
    text_top1_collapsed = text_res.collapsed_top1 if text_res else None
    text_top1_prob = text_res.collapsed_top1_prob if text_res else None

    # For image review evaluation:
    # Use top photo prediction or aggregated top-1
    img_top1_raw = image_results[0].raw_prediction if image_results else None
    img_top1_collapsed = fusion.top1 if (image_results and fusion.top1 != "UNDETERMINED") else None
    img_top1_prob = image_collapsed.get(img_top1_collapsed) if (image_collapsed and img_top1_collapsed) else None

    needs_review, review_reasons = evaluate_review_rules(
        final_label=fusion.final_label,
        top1=fusion.top1,
        model_confidence=fusion.model_confidence,
        margin=fusion.margin,
        user_label=cit_label,
        text_top1_raw=text_top1_raw,
        text_top1_collapsed=text_top1_collapsed,
        text_top1_prob=text_top1_prob,
        image_top1_raw=img_top1_raw,
        image_top1_collapsed=img_top1_collapsed,
        image_top1_prob=img_top1_prob,
        any_image_needs_review=any_img_review,
        ml_models_available=ml_succeeded,
    )

    # Step 7: Persist results in transaction with audit log
    mapped_scenario = map_final_label_to_scenario(fusion.final_label)
    models_used = {
        "text": "indicbert-disaster-api" if text_res else None,
        "image": "convnext_tiny" if image_results else None,
    }
    text_json = (
        {
            "prediction": text_res.raw_prediction,
            "confidence": text_res.confidence,
            "probabilities": text_res.raw_probabilities,
        }
        if text_res
        else None
    )
    image_json = [
        {
            "evidence_id": img.evidence_id,
            "prediction": img.raw_prediction,
            "confidence": img.confidence,
            "needs_review": img.needs_review,
            "all_probabilities": img.all_probabilities,
        }
        for img in image_results
    ]

    with db.transaction() as conn:
        conn.execute(
            """
            UPDATE incident_classifications
            SET status = %s,
                completed_at = now(),
                text_result = %s,
                image_results = %s,
                fused_probabilities = %s,
                final_label = %s,
                label_source = %s,
                model_confidence = %s,
                confidence_level = %s,
                margin = %s,
                agreement = %s,
                human_review_required = %s,
                review_reasons = %s,
                mapped_scenario = %s,
                models_used = %s,
                error_code = NULL
            WHERE classification_id = %s
            """,
            (
                status,
                Jsonb(text_json) if text_json else None,
                Jsonb(image_json) if image_json else None,
                Jsonb(fusion.fused_probabilities),
                fusion.final_label,
                fusion.label_source,
                fusion.model_confidence,
                fusion.confidence_level,
                fusion.margin,
                fusion.agreement,
                needs_review,
                review_reasons,
                mapped_scenario,
                Jsonb(models_used),
                job_id,
            ),
        )

        audit_action = "CLASSIFICATION_COMPLETED" if status == "COMPLETED" else "CLASSIFICATION_DEGRADED"
        audit.record(
            conn,
            actor=None,
            action=audit_action,
            entity_type="incident",
            entity_id=incident_id,
            result="SUCCESS",
            rationale=f"Report classified: {fusion.final_label} ({fusion.label_source}, review={needs_review})",
            after={
                "classification_id": str(job_id),
                "final_label": fusion.final_label,
                "label_source": fusion.label_source,
                "model_confidence": fusion.model_confidence,
                "human_review_required": needs_review,
                "review_reasons": review_reasons,
            },
        )

    logger.info(
        "Classification %s finished with status %s: final_label=%s (review=%s)",
        job_id,
        status,
        fusion.final_label,
        needs_review,
    )

    return {
        "classification_id": str(job_id),
        "status": status,
        "final_label": fusion.final_label,
        "label_source": fusion.label_source,
        "model_confidence": fusion.model_confidence,
        "human_review_required": needs_review,
        "review_reasons": review_reasons,
    }
