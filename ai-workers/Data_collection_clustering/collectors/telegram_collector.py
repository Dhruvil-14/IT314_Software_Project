"""
Telegram public channel collector for Disaster Coordination Hub.
Scrapes the public web view of a Telegram channel (no auth) and converts messages to Report.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup

from config import DIR_COLLECTED_DATA
from collectors.models import Report

logger = logging.getLogger("DisasterHub.TelegramCollector")

# Example public channels (add more as needed)
CHANNEL_URLS = [
    "https://t.me/s/gujarat_disaster_alerts",
    "https://t.me/s/gujarat_weather_updates",
    "https://t.me/s/gujarat_emergency",
]

class TelegramCollector:
    """Fetches recent messages from public Telegram channels.

    The scraper loads the HTML page, extracts message blocks, and creates a
    :class:`Report` for each. Raw HTML is stored under ``DIR_COLLECTED_DATA``.
    """

    def __init__(self, timeout: int = 12):
        self.timeout = timeout

    def _fetch_channel(self, url: str) -> List[Report]:
        try:
            resp = requests.get(url, timeout=self.timeout, headers={"User-Agent": "DisasterHub/0.1"})
            resp.raise_for_status()
            html = resp.text
            # Save raw HTML for audit
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fname = f"telegram_{url.split('/')[-1]}_{timestamp}.html"
            (DIR_COLLECTED_DATA / fname).write_text(html, encoding="utf-8")
        except Exception as exc:
            logger.error(f"Failed to fetch Telegram channel {url}: {exc}")
            return []

        soup = BeautifulSoup(html, "html.parser")
        reports: List[Report] = []
        # Telegram web view messages are in divs with class 'tgme_widget_message'
        for msg_div in soup.select("div.tgme_widget_message"):
            # Message text
            text_elem = msg_div.select_one("div.tgme_widget_message_text")
            text = text_elem.get_text(separator=" ", strip=True) if text_elem else ""
            # Message link (permalink)
            link_elem = msg_div.select_one("a.tgme_widget_message_date")
            link = link_elem["href"] if link_elem else url
            # Timestamp from datetime attribute (ISO) if present
            timestamp_iso = msg_div.get("data-post-date") or datetime.utcnow().isoformat()
            # Simple author extraction – Telegram shows name in 'tgme_widget_message_from'
            author_elem = msg_div.select_one("a.tgme_widget_message_from")
            author = author_elem.get_text(strip=True) if author_elem else "telegram_user"
            # Build deterministic id
            report_id = f"telegram_{link.replace('/', '_')}"
            report = Report(
                id=report_id,
                source="telegram",
                source_type="social",
                title=text[:120] + ("..." if len(text) > 120 else ""),
                text=text,
                url=link,
                timestamp=timestamp_iso,
                epoch_time=datetime.fromisoformat(timestamp_iso.rstrip('Z')).timestamp(),
                author=author,
                media_urls=[],
                is_official_alert=False,
            )
            reports.append(report)
        logger.info(f"TelegramCollector fetched {len(reports)} reports from {url}")
        return reports

    def fetch_live_reports(self) -> List[Report]:
        all_reports: List[Report] = []
        for ch in CHANNEL_URLS:
            all_reports.extend(self._fetch_channel(ch))
        logger.info(f"TelegramCollector total fetched {len(all_reports)} reports")
        return all_reports
