"""
Data models for the Disaster Coordination Hub Pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

@dataclass
class Report:
    """Standardized report representation across all pipeline stages."""
    id: str
    source: str                     # e.g., 'mastodon', 'gdelt', 'gdacs', 'imd', 'usgs'
    source_type: str                # 'social', 'news', 'official_alert'
    title: str
    text: str
    url: str
    timestamp: str                  # ISO 8601 string UTC
    epoch_time: float               # Unix timestamp
    author: str
    media_urls: List[str] = field(default_factory=list)
    local_image_paths: List[str] = field(default_factory=list)  # Local paths to downloaded image files
    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enriched fields populated by subsequent stages
    disaster_type: str = "Unclassified"
    district: Optional[str] = None
    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    is_gujarat_verified: bool = False
    
    # Classification predictions
    text_classifier_label: Optional[str] = None
    text_classifier_confidence: float = 0.0
    image_classifier_label: Optional[str] = None
    image_classifier_confidence: float = 0.0
    ensemble_disaster_type: Optional[str] = None
    
    # Clustering & Prioritization
    cluster_incident_id: Optional[str] = None
    is_official_alert: bool = False
    urgency_score: float = 0.0
    severity_level: str = "LOW"
    cleaning_notes: List[str] = field(default_factory=list)

    def to_after_cleaning_csv_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for after cleaning export (Step 2 output)."""
        return {
            "report_id": self.id,
            "source": self.source,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "district": self.district or "Unknown",
            "location_name": self.location_name or "Gujarat (General)",
            "latitude": f"{self.latitude:.4f}" if self.latitude is not None else "",
            "longitude": f"{self.longitude:.4f}" if self.longitude is not None else "",
            "is_official": self.is_official_alert,
            "title": self.title[:150].replace("\n", " ").strip(),
            "cleaned_text": self.text[:250].replace("\n", " ").strip(),
            "url": self.url,
            "media_count": len(self.media_urls),
            "is_gujarat_verified": self.is_gujarat_verified
        }

    def to_pre_cluster_csv_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for before clustering export (Step 4 output)."""
        return {
            "report_id": self.id,
            "source": self.source,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "district": self.district or "Unknown",
            "location_name": self.location_name or "Gujarat (General)",
            "latitude": f"{self.latitude:.4f}" if self.latitude is not None else "",
            "longitude": f"{self.longitude:.4f}" if self.longitude is not None else "",
            "disaster_type": self.ensemble_disaster_type or self.disaster_type,
            "text_classifier_prediction": self.text_classifier_label or "N/A",
            "text_confidence": f"{self.text_classifier_confidence:.2f}",
            "image_classifier_prediction": self.image_classifier_label or "No_Image",
            "image_confidence": f"{self.image_classifier_confidence:.2f}",
            "is_official": self.is_official_alert,
            "title": self.title[:150].replace("\n", " ").strip(),
            "text_snippet": self.text[:250].replace("\n", " ").strip(),
            "url": self.url,
            "media_count": len(self.media_urls),
            "local_images": ";".join(self.local_image_paths) if self.local_image_paths else "None"
        }

    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for final cleaned reports result (Step 7 output)."""
        return {
            "report_id": self.id,
            "source": self.source,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "district": self.district or "Unknown",
            "location_name": self.location_name or "Gujarat (General)",
            "latitude": f"{self.latitude:.4f}" if self.latitude is not None else "",
            "longitude": f"{self.longitude:.4f}" if self.longitude is not None else "",
            "disaster_type": self.ensemble_disaster_type or self.disaster_type,
            "text_classifier_prediction": self.text_classifier_label or "N/A",
            "text_confidence": f"{self.text_classifier_confidence:.2f}",
            "image_classifier_prediction": self.image_classifier_label or "No_Image",
            "image_confidence": f"{self.image_classifier_confidence:.2f}",
            "is_official": self.is_official_alert,
            "urgency_score": f"{self.urgency_score:.1f}",
            "severity_level": self.severity_level,
            "cluster_incident_id": self.cluster_incident_id or "Unclustered",
            "title": self.title[:150].replace("\n", " ").strip(),
            "text_snippet": self.text[:250].replace("\n", " ").strip(),
            "url": self.url,
            "media_count": len(self.media_urls),
            "local_images": ";".join(self.local_image_paths) if self.local_image_paths else "None"
        }

@dataclass
class IncidentSummary:
    """Grouped incident representing a cluster of related reports (after clustering output)."""
    incident_id: str
    disaster_type: str
    primary_district: str
    location_name: str
    centroid_lat: float
    centroid_lon: float
    first_reported: str
    last_reported: str
    report_count: int
    social_count: int
    news_count: int
    official_count: int
    has_official_alert: bool
    urgency_score: float
    priority_level: str              # CRITICAL, HIGH, MEDIUM, LOW
    key_evidence: List[str] = field(default_factory=list)
    recommended_actions: str = ""
    associated_report_ids: List[str] = field(default_factory=list)

    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for after clustering & incident summary export."""
        return {
            "incident_id": self.incident_id,
            "disaster_type": self.disaster_type,
            "district": self.primary_district,
            "location_name": self.location_name,
            "centroid_lat": f"{self.centroid_lat:.4f}",
            "centroid_lon": f"{self.centroid_lon:.4f}",
            "first_reported": self.first_reported,
            "last_reported": self.last_reported,
            "report_count": self.report_count,
            "social_reports": self.social_count,
            "news_reports": self.news_count,
            "official_alerts": self.official_count,
            "has_official_alert": self.has_official_alert,
            "urgency_score": f"{self.urgency_score:.1f}",
            "priority_level": self.priority_level,
            "recommended_actions": self.recommended_actions,
            "associated_report_ids": ";".join(self.associated_report_ids),
            "key_evidence_summary": " | ".join(self.key_evidence[:3]).replace("\n", " ")
        }
