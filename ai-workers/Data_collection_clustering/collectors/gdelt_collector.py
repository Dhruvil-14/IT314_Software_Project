"""
Live News & GDELT Collector for Gujarat Disasters.
Connects to GDELT 2.0 DOC API with fallback to Live Indian News Gujarat Disaster feeds.
"""

import json
import logging
import re
import urllib.parse
import urllib.request
import feedparser
from datetime import datetime, timezone
from typing import List
from collectors.models import Report
from config import GDELT_DOC_API_URL, GOOGLE_NEWS_RSS_URL

logger = logging.getLogger("DisasterHub.GDELTCollector")

class GDELTCollector:
    """Collects real-time disaster news articles related to Gujarat."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def fetch_live_reports(self) -> List[Report]:
        """Fetch news reports from GDELT DOC API, falling back to Live Regional News RSS."""
        reports = self._fetch_gdelt_doc_api()
        if not reports:
            logger.info("GDELT API empty or rate-limited. Falling back to Live Gujarat Regional News RSS feed.")
            reports = self._fetch_live_news_rss()

        logger.info(f"GDELTCollector retrieved {len(reports)} live news reports.")
        return reports

    def _fetch_gdelt_doc_api(self) -> List[Report]:
        """Query GDELT 2.0 DOC API."""
        query = 'gujarat (flood OR cyclone OR rain OR earthquake OR fire OR storm OR landslide OR disaster)'
        encoded_query = urllib.parse.quote(query)
        url = f"{GDELT_DOC_API_URL}?query={encoded_query}&mode=artlist&maxrecords=30&format=json&sort=datedesc"

        reports: List[Report] = []
        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    articles = data.get("articles", [])
                    for idx, art in enumerate(articles):
                        title = art.get("title", "")
                        url_link = art.get("url", "")
                        source_domain = art.get("domain", "GDELT News")
                        seendate = art.get("seendate", "")
                        
                        # Parse GDELT seendate format: YYYYMMDDTHHMMSSZ
                        epoch = datetime.now(timezone.utc).timestamp()
                        iso_str = datetime.now(timezone.utc).isoformat()
                        if seendate and len(seendate) >= 15:
                            try:
                                dt = datetime.strptime(seendate[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                                epoch = dt.timestamp()
                                iso_str = dt.isoformat()
                            except Exception:
                                pass

                        rep = Report(
                            id=f"gdelt_{idx}_{abs(hash(url_link)) % 1000000}",
                            source="gdelt",
                            source_type="news",
                            title=title,
                            text=f"{title}. Published by {source_domain}. Full coverage available at source URL.",
                            url=url_link,
                            timestamp=iso_str,
                            epoch_time=epoch,
                            author=source_domain,
                            media_urls=[art.get("socialimage")] if art.get("socialimage") else [],
                            raw_metadata={"language": art.get("language"), "sourcecountry": art.get("sourcecountry")}
                        )
                        reports.append(rep)
        except Exception as ex:
            logger.warning(f"GDELT DOC API request encountered: {ex}")

        return reports

    def _fetch_live_news_rss(self) -> List[Report]:
        """Fetch live disaster articles via Google News RSS for Gujarat."""
        query = "gujarat (flood OR cyclone OR earthquake OR rain OR fire OR waterlogging OR disaster OR NDRF)"
        encoded_q = urllib.parse.quote(query)
        url = f"{GOOGLE_NEWS_RSS_URL}?q={encoded_q}&hl=en-IN&gl=IN&ceid=IN:en"

        reports: List[Report] = []
        try:
            feed = feedparser.parse(url)
            for idx, entry in enumerate(feed.entries[:35]):
                title = getattr(entry, "title", "")
                link = getattr(entry, "link", "")
                summary = getattr(entry, "summary", "")
                published = getattr(entry, "published", "")

                clean_summary = re.sub(r"<[^>]+>", " ", summary)
                clean_summary = re.sub(r"\s+", " ", clean_summary).strip()
                full_text = f"{title}. {clean_summary}"

                epoch = datetime.now(timezone.utc).timestamp()
                iso_str = datetime.now(timezone.utc).isoformat()
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                        epoch = dt.timestamp()
                        iso_str = dt.isoformat()
                    except Exception:
                        pass

                source_name = entry.source.get("title", "Regional News") if hasattr(entry, "source") else "News Source"

                rep = Report(
                    id=f"news_rss_{idx}_{abs(hash(link)) % 1000000}",
                    source="gdelt_news_stream",
                    source_type="news",
                    title=title,
                    text=full_text,
                    url=link,
                    timestamp=iso_str,
                    epoch_time=epoch,
                    author=source_name,
                    media_urls=[],
                    raw_metadata={"published_raw": published}
                )
                reports.append(rep)
        except Exception as ex:
            logger.warning(f"Live regional news RSS fetch failed: {ex}")

        return reports
