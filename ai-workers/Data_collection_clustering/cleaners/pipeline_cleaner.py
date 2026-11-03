"""
Unified Cleaning and Normalization Pipeline for Step 2.
Coordinates text normalization, spam filtering, strict Gujarat geo-fencing, and deduplication.
"""

import logging
from typing import List, Dict, Any, Tuple
from collectors.models import Report
from cleaners.normalizer import clean_text
from cleaners.spam_filter import SpamFilter
from cleaners.gujarat_filter import GujaratFilter
from cleaners.deduplicator import Deduplicator

logger = logging.getLogger("DisasterHub.CleanerPipeline")

class CleanerPipeline:
    """End-to-end cleaning and filtering for Step 2."""

    def __init__(self):
        self.spam_filter = SpamFilter()
        self.gujarat_filter = GujaratFilter()
        self.deduplicator = Deduplicator()

    def process(self, raw_reports: List[Report]) -> Tuple[List[Report], Dict[str, Any]]:
        """
        Clean, normalize, verify Gujarat localization, and deduplicate reports.
        """
        stats = {
            "total_raw": len(raw_reports),
            "normalized": 0,
            "empty_or_spam_removed": 0,
            "out_of_state_removed": 0,
            "duplicates_removed": 0,
            "final_clean": 0,
        }

        stage_1_candidates: List[Report] = []

        for rep in raw_reports:
            # 1. Text normalization
            rep.title = clean_text(rep.title)
            rep.text = clean_text(rep.text)
            stats["normalized"] += 1

            # 2. Spam & relevance filtering
            is_valid, spam_reason = self.spam_filter.evaluate_report(rep)
            if not is_valid:
                rep.cleaning_notes.append(f"Rejected: {spam_reason}")
                stats["empty_or_spam_removed"] += 1
                continue

            # 3. Strict Gujarat geo-fencing and entity matching
            is_gujarat, geo_reason, district, lat, lon = self.gujarat_filter.verify_gujarat(rep)
            if not is_gujarat:
                rep.cleaning_notes.append(f"Rejected: {geo_reason}")
                stats["out_of_state_removed"] += 1
                continue

            # Enrich report with verified geographic data
            rep.is_gujarat_verified = True
            rep.district = district
            rep.location_name = district or "Gujarat"
            if rep.latitude is None and lat is not None:
                rep.latitude = lat
            if rep.longitude is None and lon is not None:
                rep.longitude = lon

            stage_1_candidates.append(rep)

        # 4. Deduplication
        clean_reports, dup_count = self.deduplicator.deduplicate(stage_1_candidates)
        stats["duplicates_removed"] = dup_count
        stats["final_clean"] = len(clean_reports)

        logger.info(
            f"Cleaning Summary: Raw={stats['total_raw']}, "
            f"Spam/Empty Removed={stats['empty_or_spam_removed']}, "
            f"Non-Gujarat Removed={stats['out_of_state_removed']}, "
            f"Duplicates Removed={stats['duplicates_removed']}, "
            f"Retained Clean={stats['final_clean']}"
        )

        return clean_reports, stats
