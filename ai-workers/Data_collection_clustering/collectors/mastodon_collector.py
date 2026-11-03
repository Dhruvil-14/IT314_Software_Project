"""
Live Mastodon Collector for Disaster Reports in Gujarat.
Queries public Mastodon instances for tag timelines and search endpoints.
"""

import json
import logging
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import List
from collectors.models import Report
from config import MASTODON_INSTANCES, MASTODON_TAGS

logger = logging.getLogger("DisasterHub.MastodonCollector")

class MastodonCollector:
    """Collects real-time public toots from Mastodon federated network."""

    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "DisasterHub-Gujarat/1.0 (Emergency Coordination; contact: research@disasterhub.org)"
        }

    def fetch_live_reports(self, limit_per_tag: int = 15) -> List[Report]:
        """Fetch posts across configured instances and tags."""
        reports: List[Report] = []
        seen_urls = set()

        for instance in MASTODON_INSTANCES:
            for tag in MASTODON_TAGS:
                url = f"{instance}/api/v1/timelines/tag/{tag}?limit={limit_per_tag}"
                try:
                    req = urllib.request.Request(url, headers=self.headers)
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        if resp.status == 200:
                            posts = json.loads(resp.read().decode("utf-8", errors="replace"))
                            for post in posts:
                                post_url = post.get("url") or post.get("uri") or ""
                                if post_url in seen_urls:
                                    continue
                                seen_urls.add(post_url)
                                
                                report = self._parse_mastodon_post(post, source_instance=instance)
                                if report:
                                    reports.append(report)
                except Exception as ex:
                    logger.debug(f"Mastodon fetch tag '{tag}' failed on {instance}: {ex}")

        # Also attempt direct instance search for Gujarat emergency keywords
        for instance in MASTODON_INSTANCES[:2]:
            search_queries = ["gujarat flood", "ahmedabad rain", "gujarat cyclone", "gujarat emergency"]
            for q in search_queries:
                search_url = f"{instance}/api/v2/search?q={urllib.parse.quote(q)}&type=statuses&limit=10"
                try:
                    req = urllib.request.Request(search_url, headers=self.headers)
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8", errors="replace"))
                            statuses = data.get("statuses", [])
                            for post in statuses:
                                post_url = post.get("url") or post.get("uri") or ""
                                if post_url in seen_urls:
                                    continue
                                seen_urls.add(post_url)
                                report = self._parse_mastodon_post(post, source_instance=instance)
                                if report:
                                    reports.append(report)
                except Exception as ex:
                    logger.debug(f"Mastodon search '{q}' failed on {instance}: {ex}")

        logger.info(f"MastodonCollector retrieved {len(reports)} live candidate posts.")
        return reports

    def _parse_mastodon_post(self, post: dict, source_instance: str) -> Report:
        """Parse raw Mastodon JSON object into standardized Report model."""
        raw_content = post.get("content", "")
        # Strip HTML tags
        clean_text = re.sub(r"<[^>]+>", " ", raw_content)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        # Parse timestamp
        created_at_str = post.get("created_at", "")
        epoch_time = 0.0
        iso_str = datetime.now(timezone.utc).isoformat()
        if created_at_str:
            try:
                # E.g. '2026-11-01T15:20:00.000Z'
                dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                epoch_time = dt.timestamp()
                iso_str = dt.isoformat()
            except Exception:
                epoch_time = datetime.now(timezone.utc).timestamp()

        # Media attachments (images)
        media_urls = []
        for media in post.get("media_attachments", []):
            if media.get("type") in ["image", "video"]:
                m_url = media.get("url") or media.get("preview_url")
                if m_url:
                    media_urls.append(m_url)

        account = post.get("account", {})
        author = account.get("username", "anonymous")
        post_id = f"mstdn_{post.get('id', hash(clean_text[:30]))}"
        url = post.get("url") or post.get("uri") or f"{source_instance}/@{author}/{post.get('id', '')}"

        # Generate a descriptive title from first sentence or text slice
        first_line = clean_text.split(".")[0] if "." in clean_text else clean_text[:80]
        title = first_line.strip() or f"Mastodon post by @{author}"

        return Report(
            id=post_id,
            source="mastodon",
            source_type="social",
            title=title,
            text=clean_text,
            url=url,
            timestamp=iso_str,
            epoch_time=epoch_time,
            author=author,
            media_urls=media_urls,
            raw_metadata={"instance": source_instance, "reblogs_count": post.get("reblogs_count", 0)}
        )
