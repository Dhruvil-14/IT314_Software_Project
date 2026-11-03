"""
Bluesky public AppView collector for Disaster Coordination Hub.
Collects public posts matching disaster‑related terms for Gujarat.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List

import requests

from config import DIR_COLLECTED_DATA
from collectors.models import Report

logger = logging.getLogger("DisasterHub.BlueskyCollector")

# Define a safe user‑agent; some services block default Python UA.
HEADERS = {
    "User-Agent": "DisasterHub/0.1 (+https://github.com/your-repo)",
    "Accept": "application/json",
}

# Search terms relevant to disaster monitoring in Gujarat
SEARCH_TERMS = ["flood", "earthquake", "cyclone", "rescue"]
# Public endpoint – no auth required (may require token in future)
BASE_URL = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"

class BlueskyCollector:
    """Fetches public Bluesky posts via the AppView search API.

    The collector saves the raw JSON response to ``DIR_COLLECTED_DATA`` for audit
    and returns a list of :class:`Report` objects compatible with the existing
    pipeline.
    """

    def __init__(self, timeout: int = 10, token: str | None = None):
        self.timeout = timeout
        self.token = token

    def _fetch_term(self, term: str) -> List[dict]:
        """Perform a single term query and return the raw ``feed`` list.

        Handles 403 responses (which indicate the public endpoint now requires a
        token) by logging a warning and returning an empty list.
        """
        params = {"q": term, "limit": 25}
        hdr = HEADERS.copy()
        if self.token:
            hdr["Authorization"] = f"Bearer {self.token}"
        try:
            resp = requests.get(BASE_URL, params=params, headers=hdr, timeout=self.timeout)
            if resp.status_code == 403:
                logger.warning(f"Bluesky term '{term}' returned 403 – likely requires auth; skipping.")
                return []
            resp.raise_for_status()
            data = resp.json()
            return data.get("feed", [])
        except requests.HTTPError as exc:
            logger.error(f"Bluesky request for term '{term}' failed: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Bluesky request for term '{term}' failed: {exc}")
            return []

    def _save_raw(self, term: str, payload: List[dict]):
        """Write the raw JSON payload to ``DIR_COLLECTED_DATA``.

        Filename pattern: ``bluesky_<term>_<YYYYMMDD_HHMMSS>.json``
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"bluesky_{term}_{timestamp}.json"
        path = DIR_COLLECTED_DATA / filename
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to write Bluesky raw file {path}: {exc}")

    def fetch_live_reports(self) -> List[Report]:
        """Collect posts for all ``SEARCH_TERMS`` and return a list of ``Report``.
        """
        all_reports: List[Report] = []
        for term in SEARCH_TERMS:
            raw_feed = self._fetch_term(term)
            if not raw_feed:
                continue
            self._save_raw(term, raw_feed)
            for entry in raw_feed:
                post = entry.get("post", {})
                record = post.get("record", {})
                text = record.get("text", "")
                created_at = record.get("createdAt") or datetime.utcnow().isoformat()
                author = post.get("author", {}).get("handle", "unknown")
                uri = post.get("uri", "")
                report_id = f"bluesky_{uri.replace('/', '_')}"
                report = Report(
                    id=report_id,
                    source="bluesky",
                    source_type="social",
                    title=text[:120] + ("..." if len(text) > 120 else ""),
                    text=text,
                    url=uri,
                    timestamp=created_at,
                    epoch_time=datetime.fromisoformat(created_at.rstrip('Z')).timestamp(),
                    author=author,
                    media_urls=[],
                    is_official_alert=False,
                )
                all_reports.append(report)
            # Respectful pause between term queries
            time.sleep(1)
        logger.info(f"BlueskyCollector fetched {len(all_reports)} reports")
        return all_reports
