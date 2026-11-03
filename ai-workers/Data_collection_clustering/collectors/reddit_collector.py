"""
Reddit RSS collector for Disaster Coordination Hub.
Fetches RSS feeds for disaster‑related subreddits and converts items to Report.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import requests
import xml.etree.ElementTree as ET

from config import DIR_COLLECTED_DATA
from collectors.models import Report

logger = logging.getLogger("DisasterHub.RedditCollector")

# Subreddits to monitor (public RSS endpoints)
SUBREDDITS = [
    "floods",
    "earthquake",
    "cyclone",
    "disasternews",
]

BASE_RSS_URL = "https://www.reddit.com/r/{sub}/.rss"

class RedditCollector:
    """Collects posts from Reddit subreddits via their public RSS feeds.

    Raw XML is saved under ``DIR_COLLECTED_DATA`` and each item is transformed
    into a :class:`Report`. Only posts that mention Gujarat (or its districts)
    are retained downstream – that filtering lives in later pipeline stages.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def _fetch_rss(self, subreddit: str) -> List[ET.Element]:
        url = BASE_RSS_URL.format(sub=subreddit)
        try:
            resp = requests.get(url, timeout=self.timeout, headers={"User-Agent": "DisasterHub/0.1"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            # Reddit RSS uses the "item" tag under "channel"
            items = root.findall(".//item")
            return items
        except Exception as exc:
            logger.error(f"Failed to fetch Reddit RSS for r/{subreddit}: {exc}")
            return []

    def _save_raw(self, subreddit: str, xml_content: str):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"reddit_{subreddit}_{timestamp}.xml"
        path = DIR_COLLECTED_DATA / filename
        try:
            path.write_text(xml_content, encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Unable to write Reddit raw file {path}: {exc}")

    def fetch_live_reports(self) -> List[Report]:
        all_reports: List[Report] = []
        for sub in SUBREDDITS:
            items = self._fetch_rss(sub)
            if not items:
                continue
            # Save raw XML string for auditability
            try:
                raw_xml = requests.get(BASE_RSS_URL.format(sub=sub), timeout=self.timeout).text
                self._save_raw(sub, raw_xml)
            except Exception:
                pass
            for item in items:
                title = item.findtext("title", "")
                link = item.findtext("link", "")
                description = item.findtext("description", "")
                pub_date = item.findtext("pubDate")
                # Reddit RSS does not include author directly; use title prefix if present.
                author = "reddit_user"
                # Build a deterministic id from link
                report_id = f"reddit_{link.replace('/', '_')}"
                # Convert pubDate to ISO8601 (simple approach)
                try:
                    dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                    iso_ts = dt.isoformat()
                except Exception:
                    iso_ts = datetime.utcnow().isoformat()
                report = Report(
                    id=report_id,
                    source="reddit",
                    source_type="social",
                    title=title,
                    text=description,
                    url=link,
                    timestamp=iso_ts,
                    epoch_time=datetime.fromisoformat(iso_ts.rstrip('Z')).timestamp(),
                    author=author,
                    media_urls=[],
                    is_official_alert=False,
                )
                all_reports.append(report)
        logger.info(f"RedditCollector fetched {len(all_reports)} reports")
        return all_reports
