"""
Spatio-Temporal Clustering Engine for Step 5.
Groups reports describing the same incident at the same location within a time window.
"""

import math
from datetime import datetime, timezone
from typing import List, Tuple, Dict
from collectors.models import Report, IncidentSummary
from config import SPATIAL_CLUSTER_RADIUS_KM, TEMPORAL_CLUSTER_HOURS

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in kilometers between two points on Earth."""
    r_earth = 6371.0  # Radius of Earth in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(d_lon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth * c

class SpatioTemporalClusterer:
    """Groups related disaster reports into cohesive incident clusters."""

    def __init__(self, max_distance_km: float = SPATIAL_CLUSTER_RADIUS_KM,
                 max_hours_diff: float = TEMPORAL_CLUSTER_HOURS):
        self.max_distance_km = max_distance_km
        self.max_seconds_diff = max_hours_diff * 3600.0

    def cluster(self, reports: List[Report]) -> Tuple[List[Report], List[IncidentSummary]]:
        """
        Group reports into incidents using Connected Components clustering.
        Returns: (updated_reports_with_cluster_id, incident_summaries)
        """
        n = len(reports)
        if n == 0:
            return [], []

        # Build adjacency graph
        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if self._are_reports_related(reports[i], reports[j]):
                    adj[i].append(j)
                    adj[j].append(i)

        # Connected components via BFS
        visited = set()
        components: List[List[int]] = []

        for i in range(n):
            if i not in visited:
                comp = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(comp)

        # Generate incident summaries and assign incident IDs to reports
        incident_summaries: List[IncidentSummary] = []
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")

        for idx, comp_indices in enumerate(components, 1):
            cluster_reports = [reports[i] for i in comp_indices]
            incident_id = f"INC-GUJ-{today_str}-{idx:03d}"

            # Calculate centroid coordinates
            valid_coords = [(r.latitude, r.longitude) for r in cluster_reports if r.latitude is not None and r.longitude is not None]
            if valid_coords:
                centroid_lat = sum(c[0] for c in valid_coords) / len(valid_coords)
                centroid_lon = sum(c[1] for c in valid_coords) / len(valid_coords)
            else:
                centroid_lat, centroid_lon = 23.0225, 72.5714

            # Dominant disaster type
            disaster_counts: Dict[str, int] = {}
            for r in cluster_reports:
                dtype = r.ensemble_disaster_type or r.disaster_type or "Emergency"
                disaster_counts[dtype] = disaster_counts.get(dtype, 0) + 1
            primary_disaster = max(disaster_counts, key=disaster_counts.get)

            # District and location naming
            district_counts: Dict[str, int] = {}
            local_places = []
            for r in cluster_reports:
                if r.district:
                    district_counts[r.district] = district_counts.get(r.district, 0) + 1
                if r.location_name and r.location_name not in local_places:
                    local_places.append(r.location_name)
            primary_district = max(district_counts, key=district_counts.get) if district_counts else "Gujarat"
            location_display = ", ".join(local_places[:2]) if local_places else primary_district

            # Source breakdowns
            social_count = sum(1 for r in cluster_reports if r.source_type == "social")
            news_count = sum(1 for r in cluster_reports if r.source_type == "news")
            official_count = sum(1 for r in cluster_reports if r.source_type == "official_alert" or r.is_official_alert)
            has_official = official_count > 0

            # Time range
            epochs = [r.epoch_time for r in cluster_reports if r.epoch_time > 0]
            first_rep = datetime.fromtimestamp(min(epochs), tz=timezone.utc).isoformat() if epochs else datetime.now(timezone.utc).isoformat()
            last_rep = datetime.fromtimestamp(max(epochs), tz=timezone.utc).isoformat() if epochs else datetime.now(timezone.utc).isoformat()

            # Key evidence snippets
            key_evidence = [r.title for r in cluster_reports[:3]]

            # Update reports with cluster ID
            for r in cluster_reports:
                r.cluster_incident_id = incident_id

            summary = IncidentSummary(
                incident_id=incident_id,
                disaster_type=primary_disaster,
                primary_district=primary_district,
                location_name=location_display,
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                first_reported=first_rep,
                last_reported=last_rep,
                report_count=len(cluster_reports),
                social_count=social_count,
                news_count=news_count,
                official_count=official_count,
                has_official_alert=has_official,
                urgency_score=0.0,  # Will be populated in Step 6
                priority_level="LOW",
                key_evidence=key_evidence,
                associated_report_ids=[r.id for r in cluster_reports]
            )
            incident_summaries.append(summary)

        return reports, incident_summaries

    def _are_reports_related(self, r1: Report, r2: Report) -> bool:
        """Evaluate if two reports describe the same event."""
        # 1. Check disaster type compatibility
        d1 = (r1.ensemble_disaster_type or r1.disaster_type or "").lower()
        d2 = (r2.ensemble_disaster_type or r2.disaster_type or "").lower()
        if d1 != d2:
            # Allow Flood and Heavy_Rain overlap
            compatible_pairs = [("flood", "heavy_rain_waterlogging"), ("heavy_rain_waterlogging", "flood")]
            if (d1, d2) not in compatible_pairs:
                return False

        # 2. Check temporal proximity
        if abs(r1.epoch_time - r2.epoch_time) > self.max_seconds_diff:
            return False

        # 3. Check spatial proximity
        if r1.latitude is not None and r1.longitude is not None and r2.latitude is not None and r2.longitude is not None:
            dist_km = haversine_distance_km(r1.latitude, r1.longitude, r2.latitude, r2.longitude)
            if dist_km <= self.max_distance_km:
                return True
            else:
                return False

        # 4. Fallback: match by common district
        if r1.district and r2.district and r1.district == r2.district and r1.district != "Gujarat (Statewide)":
            return True

        return False
