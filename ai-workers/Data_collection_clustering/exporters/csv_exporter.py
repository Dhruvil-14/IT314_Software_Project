"""
CSV Exporter for Step 7 and Data Partitioning.
Partitions data into structured directories inside data/:
  1. data/after_cleaning/reports_after_cleaning.csv       (Cleaned & filtered raw data)
  2. data/before_clustering/reports_before_clustering.csv (Extracted & classified data)
  3. data/after_clustering/incidents_after_clustering.csv  (Grouped incident clusters)
  4. data/results/cleaned_reports.csv & incident_summaries.csv (Final deliverables)
"""

import csv
import logging
from pathlib import Path
from typing import List, Set, Tuple
from collectors.models import Report, IncidentSummary
from config import (
    DIR_AFTER_CLEANING, DIR_BEFORE_CLUSTERING, DIR_AFTER_CLUSTERING, DIR_RESULTS,
    OUTPUT_AFTER_CLEANING, OUTPUT_BEFORE_CLUSTERING, OUTPUT_AFTER_CLUSTERING,
    OUTPUT_CLEANED_REPORTS, OUTPUT_INCIDENT_SUMMARIES
)

logger = logging.getLogger("DisasterHub.CSVExporter")

class CSVExporter:
    """Manages writing data across all partitioned data directories."""

    def __init__(self,
                 after_cleaning_path: Path = OUTPUT_AFTER_CLEANING,
                 before_path: Path = OUTPUT_BEFORE_CLUSTERING,
                 after_path: Path = OUTPUT_AFTER_CLUSTERING,
                 results_reports_path: Path = OUTPUT_CLEANED_REPORTS,
                 results_incidents_path: Path = OUTPUT_INCIDENT_SUMMARIES):
        self.after_cleaning_path = Path(after_cleaning_path)
        self.before_path = Path(before_path)
        self.after_path = Path(after_path)
        self.results_reports_path = Path(results_reports_path)
        self.results_incidents_path = Path(results_incidents_path)

        # Legacy alias for backward compatibility
        self.reports_path = self.results_reports_path
        self.incidents_path = self.results_incidents_path

        # Ensure all partition directories exist
        DIR_AFTER_CLEANING.mkdir(parents=True, exist_ok=True)
        DIR_BEFORE_CLUSTERING.mkdir(parents=True, exist_ok=True)
        DIR_AFTER_CLUSTERING.mkdir(parents=True, exist_ok=True)
        DIR_RESULTS.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PART 1: Storing Data AFTER Cleaning (Step 2 output)
    # =========================================================================
    def export_after_cleaning(self, reports: List[Report], append: bool = True) -> Path:
        """
        Write or append individual cleaned reports immediately after Step 2.
        Contains verified Gujarat, spam-filtered, deduplicated normalized records.
        """
        if not reports:
            return self.after_cleaning_path

        fieldnames = list(reports[0].to_after_cleaning_csv_dict().keys())
        file_exists = self.after_cleaning_path.exists() and self.after_cleaning_path.stat().st_size > 0

        existing_ids = self._get_existing_ids(self.after_cleaning_path) if (append and file_exists) else set()
        records_to_write = [r for r in reports if r.id not in existing_ids]

        if not records_to_write and append and file_exists:
            return self.after_cleaning_path

        mode = "a" if (append and file_exists) else "w"
        with open(self.after_cleaning_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            if mode == "w":
                writer.writeheader()
            for r in (records_to_write if append else reports):
                writer.writerow(r.to_after_cleaning_csv_dict())

        logger.info(f"[After Cleaning] Saved {len(records_to_write if append else reports)} records to {self.after_cleaning_path}")
        return self.after_cleaning_path

    # =========================================================================
    # PART 2: Storing Data BEFORE Clustering (Step 4 output)
    # =========================================================================
    def export_before_clustering(self, reports: List[Report], append: bool = True) -> Path:
        """
        Write or append individual cleaned & classified reports before clustering.
        Contains raw features, classifications, coordinates, and extracted places.
        """
        if not reports:
            return self.before_path

        fieldnames = list(reports[0].to_pre_cluster_csv_dict().keys())
        file_exists = self.before_path.exists() and self.before_path.stat().st_size > 0

        existing_ids = self._get_existing_ids(self.before_path) if (append and file_exists) else set()
        records_to_write = [r for r in reports if r.id not in existing_ids]

        if not records_to_write and append and file_exists:
            return self.before_path

        mode = "a" if (append and file_exists) else "w"
        with open(self.before_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            if mode == "w":
                writer.writeheader()
            for r in (records_to_write if append else reports):
                writer.writerow(r.to_pre_cluster_csv_dict())

        logger.info(f"[Before Clustering] Saved {len(records_to_write if append else reports)} records to {self.before_path}")
        return self.before_path

    # =========================================================================
    # PART 3: Storing Data AFTER Clustering (Step 5 output)
    # =========================================================================
    def export_after_clustering(self, incidents: List[IncidentSummary]) -> Path:
        """
        Write grouped incident clusters formed after spatio-temporal clustering.
        Contains cluster centroids, report counts, urgency scores, and member IDs.
        """
        if not incidents:
            with open(self.after_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["incident_id", "disaster_type", "district", "urgency_score", "priority_level"])
            return self.after_path

        fieldnames = list(incidents[0].to_csv_dict().keys())
        with open(self.after_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for inc in incidents:
                writer.writerow(inc.to_csv_dict())

        logger.info(f"[After Clustering] Saved {len(incidents)} incident clusters to {self.after_path}")
        return self.after_path

    # =========================================================================
    # PART 4: Storing Final RESULTS (Step 6 & 7 outputs)
    # =========================================================================
    def export_results(self, reports: List[Report], incidents: List[IncidentSummary],
                       append: bool = True) -> Tuple[Path, Path]:
        """
        Write final results:
          - data/results/cleaned_reports.csv (All cleaned reports with assigned cluster IDs & urgency)
          - data/results/incident_summaries.csv (Consolidated incident overview & action recommendations)
        """
        # 1. Cleaned Reports Result
        if reports:
            fieldnames = list(reports[0].to_csv_dict().keys())
            file_exists = self.results_reports_path.exists() and self.results_reports_path.stat().st_size > 0
            existing_ids = self._get_existing_ids(self.results_reports_path) if (append and file_exists) else set()
            records_to_write = [r for r in reports if r.id not in existing_ids]

            if records_to_write or not file_exists or not append:
                mode = "a" if (append and file_exists) else "w"
                with open(self.results_reports_path, mode, newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                    if mode == "w":
                        writer.writeheader()
                    for r in (records_to_write if append else reports):
                        writer.writerow(r.to_csv_dict())
                logger.info(f"[Final Result] Saved {len(records_to_write if append else reports)} reports to {self.results_reports_path}")

        # 2. Incident Summaries Result
        if incidents:
            fieldnames = list(incidents[0].to_csv_dict().keys())
            with open(self.results_incidents_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                for inc in incidents:
                    writer.writerow(inc.to_csv_dict())
            logger.info(f"[Final Result] Saved {len(incidents)} incident summaries to {self.results_incidents_path}")

        return self.results_reports_path, self.results_incidents_path

    def export(self, reports: List[Report], incidents: List[IncidentSummary],
               append: bool = True) -> Tuple[Path, Path]:
        """Backward compatibility wrapper."""
        self.export_after_clustering(incidents)
        return self.export_results(reports, incidents, append=append)

    def load_existing_reports(self) -> List[Report]:
        """Read back previously stored reports from results/cleaned_reports.csv."""
        target = self.results_reports_path if self.results_reports_path.exists() else self.reports_path
        if not target.exists():
            return []

        loaded: List[Report] = []
        try:
            with open(target, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lat_str = row.get("latitude", "")
                    lon_str = row.get("longitude", "")
                    lat = float(lat_str) if lat_str else None
                    lon = float(lon_str) if lon_str else None
                    
                    rep = Report(
                        id=row.get("report_id", ""),
                        source=row.get("source", ""),
                        source_type=row.get("source_type", "news"),
                        title=row.get("title", ""),
                        text=row.get("text_snippet", ""),
                        url=row.get("url", ""),
                        timestamp=row.get("timestamp", ""),
                        epoch_time=0.0,
                        author="",
                        district=row.get("district"),
                        location_name=row.get("location_name"),
                        latitude=lat,
                        longitude=lon,
                        disaster_type=row.get("disaster_type", "Emergency"),
                        ensemble_disaster_type=row.get("disaster_type"),
                        cluster_incident_id=row.get("cluster_incident_id"),
                        is_official_alert=(row.get("is_official") == "True"),
                        urgency_score=float(row.get("urgency_score", 0.0)),
                        severity_level=row.get("severity_level", "LOW")
                    )
                    loaded.append(rep)
        except Exception as ex:
            logger.warning(f"Error loading existing reports: {ex}")

        return loaded

    def _get_existing_ids(self, csv_file: Path) -> Set[str]:
        """Extract existing report IDs to prevent duplicates."""
        existing = set()
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = row.get("report_id")
                    if rid:
                        existing.add(rid)
        except Exception:
            pass
        return existing
