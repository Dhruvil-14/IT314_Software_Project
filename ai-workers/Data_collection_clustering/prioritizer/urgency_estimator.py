"""
Prioritization and Urgency Estimation Engine for Step 6.
Estimates urgency and severity from multi-source evidence, official alerts, and casualty signals.
"""

import re
from datetime import datetime, timezone
from typing import List, Tuple
from collectors.models import Report, IncidentSummary
from config import PRIORITY_WEIGHTS

HIGH_SEVERITY_KEYWORDS = [
    r"\bcasualt(y|ies)\b", r"\bdeath toll\b", r"\bdead\b", r"\bkilled\b",
    r"\bdrowned\b", r"\btrapped\b", r"\bstranded\b", r"\bunder debris\b",
    r"\bevacuat(ed|ion)\b", r"\bdam breach\b", r"\bdanger mark\b", r"\bexplos(ion|ive)\b",
    r"\btoxic\b", r"\bmassive blaze\b"
]

MODERATE_SEVERITY_KEYWORDS = [
    r"\bwaterlogg(ed|ing)\b", r"\bsubmerged\b", r"\broads blocked\b",
    r"\btrees uprooted\b", r"\bpower out(age)?\b", r"\bgusts\b", r"\bheavy rain\b",
    r"\brubber boats\b", r"\bfire tenders\b"
]

class UrgencyEstimator:
    """Calculates granular urgency scores (0-100) and priority categories."""

    def __init__(self):
        self.weights = PRIORITY_WEIGHTS

    def prioritize_incidents(self, incidents: List[IncidentSummary],
                             reports: List[Report]) -> Tuple[List[IncidentSummary], List[Report]]:
        """
        Evaluate urgency for all incident clusters and update constituent reports.
        """
        report_map = {r.id: r for r in reports}

        for inc in incidents:
            cluster_reports = [report_map[rid] for rid in inc.associated_report_ids if rid in report_map]
            
            score, severity_notes = self._calculate_cluster_urgency(inc, cluster_reports)
            inc.urgency_score = round(score, 1)

            # Assign Priority Level
            if score >= 75.0:
                inc.priority_level = "CRITICAL"
                inc.recommended_actions = (
                    "IMMEDIATE EMERGENCY MOBILIZATION: Deploy NDRF/SDRF battalions, "
                    "enforce mandatory evacuation in low-lying zones, establish medical relief camp."
                )
            elif score >= 50.0:
                inc.priority_level = "HIGH"
                inc.recommended_actions = (
                    "URGENT DEPLOYMENT: Deploy Municipal Fire & Rescue teams, dispatch high-capacity "
                    "dewatering pumps, issue localized road diversions, open relief shelters."
                )
            elif score >= 25.0:
                inc.priority_level = "MEDIUM"
                inc.recommended_actions = (
                    "MONITOR & STANDBY: Alert local ward disaster cells, monitor river/reservoir gauges, "
                    "issue public travel cautionary advisory."
                )
            else:
                inc.priority_level = "LOW"
                inc.recommended_actions = (
                    "ROUTINE MONITORING: Maintain active surveillance on automated sensor feeds and social reports."
                )

            # Synchronize urgency and severity back to each report in this incident
            for r in cluster_reports:
                r.urgency_score = inc.urgency_score
                r.severity_level = inc.priority_level

        # Sort incidents descending by urgency score
        incidents.sort(key=lambda x: x.urgency_score, reverse=True)
        return incidents, reports

    def _calculate_cluster_urgency(self, inc: IncidentSummary,
                                   cluster_reports: List[Report]) -> Tuple[float, List[str]]:
        """Compute composite priority score based on evidence, official alerts, and severity."""
        total_score = 0.0
        notes = []

        # 1. Official Alert Contribution (Up to 40 pts)
        combined_txt = " ".join([f"{r.title} {r.text}" for r in cluster_reports]).lower()
        if inc.has_official_alert or inc.official_count > 0:
            if "red alert" in combined_txt or "critical" in combined_txt:
                total_score += 40.0
                notes.append("Official RED/CRITICAL Emergency Alert active")
            else:
                total_score += self.weights["official_alert"]
                notes.append("Official Emergency Alert active")
        else:
            if "red alert" in combined_txt:
                total_score += 30.0
                notes.append("Mentions Red Alert")
            elif "orange alert" in combined_txt:
                total_score += 20.0
                notes.append("Mentions Orange Alert")

        # 2. Severity Keywords Contribution (Up to 30 pts)
        combined_text = " ".join([f"{r.title} {r.text}" for r in cluster_reports]).lower()
        high_sev_hits = sum(1 for kw in HIGH_SEVERITY_KEYWORDS if re.search(kw, combined_text))
        mod_sev_hits = sum(1 for kw in MODERATE_SEVERITY_KEYWORDS if re.search(kw, combined_text))

        sev_score = min(30.0, (high_sev_hits * 10.0) + (mod_sev_hits * 4.0))
        total_score += sev_score
        if high_sev_hits > 0:
            notes.append(f"{high_sev_hits} critical casualty/damage indicators")

        # 3. Evidence Count and Cross-Source Multiplier (Up to 20 pts)
        count = inc.report_count
        count_score = min(12.0, count * 3.0)
        
        # Cross-source diversity bonus (social + news + official)
        source_types = set(r.source_type for r in cluster_reports)
        diversity_bonus = len(source_types) * 4.0  # Up to 8 or 12 pts
        evidence_score = min(20.0, count_score + diversity_bonus)
        total_score += evidence_score

        # 4. Recency Score (Up to 10 pts)
        now_epoch = datetime.now(timezone.utc).timestamp()
        epochs = [r.epoch_time for r in cluster_reports if r.epoch_time > 0]
        if epochs:
            most_recent_diff_hours = (now_epoch - max(epochs)) / 3600.0
            if most_recent_diff_hours <= 3.0:
                recency_score = 10.0
            elif most_recent_diff_hours <= 6.0:
                recency_score = 7.0
            elif most_recent_diff_hours <= 12.0:
                recency_score = 5.0
            else:
                recency_score = 2.0
            total_score += recency_score

        # 5. Visual Evidence / Image Confirmation (Up to 5 pts)
        has_media = any(len(r.media_urls) > 0 for r in cluster_reports)
        if has_media:
            total_score += self.weights["has_image"]
            notes.append("Visual media verified")

        final_score = max(5.0, min(100.0, total_score))
        return final_score, notes
