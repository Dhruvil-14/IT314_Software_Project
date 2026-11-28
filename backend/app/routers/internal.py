"""Token-protected cron endpoint for scheduled ingestion."""

from __future__ import annotations

import hmac
import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.services.ingestion import run_ingestion

logger = logging.getLogger("drch.ingest")
router = APIRouter(prefix="/api/v1/internal", tags=["internal"])


class IngestRunResponse(BaseModel):
    run_id: str
    source: str
    items_fetched: int
    items_stored: int
    status: str
    message: str | None = None


class ClassificationSweepResponse(BaseModel):
    reset_count: int
    processed_count: int
    message: str = "Classification sweep completed successfully"


@router.post("/classification/sweep", response_model=ClassificationSweepResponse)
async def sweep_classification(
    limit: Annotated[int, Query(ge=1, le=200, description="Max jobs to process")] = 50,
    x_sweep_token: Annotated[str | None, Header(alias="X-Classification-Sweep-Token")] = None,
    authorization: Annotated[str | None, Header()] = None,
):
    """Run stale job sweep and process pending classification jobs; accepts X-Classification-Sweep-Token or Bearer token."""
    expected_token = get_settings().classifier_sweep_token
    provided_token = x_sweep_token.strip() if x_sweep_token else None
    if not provided_token and authorization and authorization.lower().startswith("bearer "):
        provided_token = authorization[7:].strip()
    if not expected_token or not provided_token or not hmac.compare_digest(provided_token, expected_token):
        logger.warning("Rejected unauthenticated or invalid classification sweep trigger attempt")
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing sweep token")

    def _execute_sweep(batch_limit: int) -> tuple[int, int]:
        from app.services.classification.worker import run_classification, sweep_stale_classifications

        swept = sweep_stale_classifications()
        processed = 0
        while processed < batch_limit:
            res = run_classification()
            if not res or res.get("status") == "RETRY_SCHEDULED":
                break
            processed += 1
        return swept, processed

    swept, processed = await run_in_threadpool(_execute_sweep, limit)
    return ClassificationSweepResponse(
        reset_count=swept,
        processed_count=processed,
        message=f"Swept {swept} stale running jobs, processed {processed} classifications.",
    )


@router.post("/ingest/run", response_model=IngestRunResponse | list[IngestRunResponse])
async def run_ingest(
    source: Annotated[str, Query(description="Ingestion source name or ALL")] = "ALL",
    x_ingest_token: Annotated[str | None, Header(alias="X-Ingest-Token")] = None,
    authorization: Annotated[str | None, Header()] = None,
):
    """Run all feeds or a named source; accepts X-Ingest-Token or a bearer token."""
    expected_token = get_settings().ingest_token
    provided_token = x_ingest_token.strip() if x_ingest_token else None
    if not provided_token and authorization and authorization.lower().startswith("bearer "):
        provided_token = authorization[7:].strip()
    if not expected_token or not provided_token or not hmac.compare_digest(provided_token, expected_token):
        logger.warning("Rejected unauthenticated or invalid ingestion trigger attempt")
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing ingestion token")

    results = await run_in_threadpool(run_ingestion, source, get_settings())
    if source.strip().upper() != "ALL" and results:
        return results[0]
    return results
