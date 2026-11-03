"""
Deduplication Engine for Reports.
Performs exact hash deduplication and near-duplicate fuzzy matching.
"""

import hashlib
from typing import List, Tuple
from collectors.models import Report

class Deduplicator:
    """Removes exact and near-duplicate reports."""

    def __init__(self, jaccard_threshold: float = 0.82):
        self.jaccard_threshold = jaccard_threshold

    def deduplicate(self, reports: List[Report]) -> Tuple[List[Report], int]:
        """
        Deduplicate reports list. Returns (unique_reports, num_duplicates_removed).
        """
        unique_reports: List[Report] = []
        seen_hashes = set()
        seen_token_sets = []
        dup_count = 0

        for rep in reports:
            # 1. Exact text hash check
            norm_content = rep.text.lower().strip()
            content_hash = hashlib.sha256(norm_content.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                dup_count += 1
                continue

            # 2. Token set Jaccard similarity check (for near duplicates/syndicated articles)
            tokens = set(norm_content.split())
            if len(tokens) >= 5:
                is_near_dup = False
                for existing_tokens in seen_token_sets:
                    intersection = len(tokens.intersection(existing_tokens))
                    union = len(tokens.union(existing_tokens))
                    if union > 0 and (intersection / union) >= self.jaccard_threshold:
                        is_near_dup = True
                        break
                if is_near_dup:
                    dup_count += 1
                    continue

            seen_hashes.add(content_hash)
            if len(tokens) >= 5:
                seen_token_sets.append(tokens)
            unique_reports.append(rep)

        return unique_reports, dup_count
