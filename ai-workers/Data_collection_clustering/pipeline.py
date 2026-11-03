"""
Master Disaster Coordination Pipeline for Gujarat (Sprint 2).
Executes all 7 workflow stages from live ingestion to prioritized CSV output.
Supports 4 structured data partitions inside the data directory:
  1. data/after_cleaning/reports_after_cleaning.csv
  2. data/before_clustering/reports_before_clustering.csv
  3. data/after_clustering/incidents_after_clustering.csv
  4. data/results/cleaned_reports.csv & incident_summaries.csv
"""

import logging
import sys
import time
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DisasterHub.Pipeline")

from collectors.models import Report, IncidentSummary
from collectors.aggregator import DataAggregator
from collectors.image_downloader import ImageDownloader
from collectors.state_tracker import StateTracker
from cleaners.pipeline_cleaner import CleanerPipeline
from extractors.information_extractor import InformationExtractor
from classifiers.base import BaseTextClassifier, BaseImageClassifier
from classifiers.ensemble import DisasterEnsembleClassifier
from clustering.spatiotemporal_clusterer import SpatioTemporalClusterer
from prioritizer.urgency_estimator import UrgencyEstimator
from exporters.csv_exporter import CSVExporter

class DisasterHubPipeline:
    """
    Sprint 2 Complete Pipeline for Gujarat Disaster Coordination Hub.
    
    Stages:
      1. Collect live data (Mastodon, GDELT/News, Official Alerts)
      2. Clean and normalize (Remove spam, empty, out-of-state, duplicates)
         -> Saves to data/after_cleaning/
      3. Extract information (Disaster type, places, timestamps, coordinates)
      4. Classify disaster type (Text classifier + Image classifier ensemble)
         -> Saves to data/before_clustering/
      5. Cluster related reports (Spatio-temporal grouping into incidents)
         -> Saves to data/after_clustering/
      6. Prioritize incidents (Estimate urgency from official alerts & severity)
      7. Save results (Export final cleaned reports & incident summaries to data/results/)
    """

    def __init__(self, text_classifier: Optional[BaseTextClassifier] = None,
                 image_classifier: Optional[BaseImageClassifier] = None):
        logger.info("Initializing Disaster Coordination Hub Pipeline (Gujarat Scope)...")
        
        # Pipeline components
        self.aggregator = DataAggregator()
        self.image_downloader = ImageDownloader()
        self.state_tracker = StateTracker()
        self.cleaner = CleanerPipeline()
        self.extractor = InformationExtractor()
        self.ensemble_classifier = DisasterEnsembleClassifier(
            text_classifier=text_classifier,
            image_classifier=image_classifier
        )
        self.clusterer = SpatioTemporalClusterer()
        self.prioritizer = UrgencyEstimator()
        self.exporter = CSVExporter()

    def set_text_classifier(self, classifier: BaseTextClassifier):
        """Hook to plug in user's custom trained text model."""
        self.ensemble_classifier.set_text_classifier(classifier)

    def set_image_classifier(self, classifier: BaseImageClassifier):
        """Hook to plug in user's custom trained vision model."""
        self.ensemble_classifier.set_image_classifier(classifier)

    def run(self, use_live_feeds: bool = True, download_images: bool = True,
            incremental: bool = True) -> Dict[str, Any]:
        """
        Execute the end-to-end 7-step pipeline.
        
        Args:
            use_live_feeds: True to fetch live internet endpoints; False for seed data.
            download_images: True to download image attachments locally to data/images/.
            incremental: True to only collect data since last run and append to CSV;
                         False to overwrite CSVs and restart from scratch.
        """
        start_time = time.time()
        logger.info("=" * 70)
        logger.info("STARTING SPRINT 2 PIPELINE EXECUTION")
        logger.info(f"Target Jurisdiction: GUJARAT ONLY | Mode: {'LIVE ONLINE' if use_live_feeds else 'OFFLINE SEED'}")
        logger.info(f"Execution Strategy: {'INCREMENTAL (APPEND NEW DATA)' if incremental else 'FRESH RUN (OVERWRITE)'}")
        logger.info("=" * 70)

        if not incremental:
            self.state_tracker.reset_state()

        # STAGE 1: Collect live data
        logger.info("\n--- [STAGE 1/7] Collecting Live Data ---")
        raw_reports = self.aggregator.collect_all(use_live_feeds=use_live_feeds)
        logger.info(f"-> Raw reports collected from endpoints: {len(raw_reports)}")

        # Incremental Watermarking: Filter out previously collected reports
        if incremental and self.state_tracker.last_run_epoch > 0:
            logger.info("Applying watermark: checking timestamp and IDs against previous runs...")
            raw_reports = self.state_tracker.filter_new_reports(raw_reports)
            logger.info(f"-> New candidate reports since last collection: {len(raw_reports)}")

        if not raw_reports:
            logger.info("No new reports discovered since last run. Everything is up to date!")
            existing_reports = self.exporter.load_existing_reports()
            _, incidents = self.clusterer.cluster(existing_reports)
            prioritized, _ = self.prioritizer.prioritize_incidents(incidents, existing_reports)
            return {
                "status": "up_to_date",
                "new_reports_count": 0,
                "cleaned_reports_count": 0,
                "total_reports_count": len(existing_reports),
                "incidents_count": len(prioritized),
                "after_cleaning_csv": str(self.exporter.after_cleaning_path),
                "before_clustering_csv": str(self.exporter.before_path),
                "after_clustering_csv": str(self.exporter.after_path),
                "results_reports_csv": str(self.exporter.results_reports_path),
                "results_incidents_csv": str(self.exporter.results_incidents_path),
                "reports_csv_path": str(self.exporter.results_reports_path),
                "incidents_csv_path": str(self.exporter.results_incidents_path),
                "incidents": prioritized,
                "reports": existing_reports
            }

        # STAGE 2: Clean and normalize
        logger.info("\n--- [STAGE 2/7] Cleaning & Normalizing Data ---")
        cleaned_reports, clean_stats = self.cleaner.process(raw_reports)
        logger.info(f"-> Filtered: {clean_stats['empty_or_spam_removed']} spam/empty, "
                    f"{clean_stats['out_of_state_removed']} non-Gujarat, "
                    f"{clean_stats['duplicates_removed']} duplicates.")
        logger.info(f"-> Clean reports retained: {len(cleaned_reports)}")

        if not cleaned_reports:
            logger.warning("No reports remained after cleaning.")
            return {"status": "empty", "elapsed_seconds": time.time() - start_time}

        # =====================================================================
        # PART 1: Storing Data AFTER CLEANING
        # =====================================================================
        logger.info("\n--- [DATA PARTITION 1/4] Saving Data AFTER Cleaning ---")
        after_cleaning_csv = self.exporter.export_after_cleaning(cleaned_reports, append=incremental)
        logger.info(f"-> Data After Cleaning: {after_cleaning_csv}")

        # STAGE 3: Extract information
        logger.info("\n--- [STAGE 3/7] Extracting Information ---")
        for rep in cleaned_reports:
            self.extractor.extract(rep)
        logger.info(f"-> Extracted entities, places, timestamps and coordinates for {len(cleaned_reports)} reports.")

        # STAGE 3.5: Download Attached Images Locally
        if download_images:
            logger.info("\n--- [MEDIA INGESTION] Downloading Attached Images Locally ---")
            downloaded = self.image_downloader.download_reports_images(cleaned_reports)
            logger.info(f"-> Downloaded {downloaded} disaster images to local storage for vision classifier.")

        # STAGE 4: Classify disaster type
        logger.info("\n--- [STAGE 4/7] Classifying Disaster Type (Multi-modal) ---")
        for rep in cleaned_reports:
            self.ensemble_classifier.classify_report(rep)
        logger.info(f"-> Classified {len(cleaned_reports)} reports with Text & Image Classifiers.")

        # =====================================================================
        # PART 2: Storing Data BEFORE CLUSTERING
        # =====================================================================
        logger.info("\n--- [DATA PARTITION 2/4] Saving Data BEFORE Clustering ---")
        before_csv = self.exporter.export_before_clustering(cleaned_reports, append=incremental)
        logger.info(f"-> Data Before Clustering: {before_csv}")

        # STAGE 5: Cluster related reports
        logger.info("\n--- [STAGE 5/7] Spatio-Temporal Incident Clustering ---")
        if incremental and self.exporter.results_reports_path.exists():
            historical_reports = self.exporter.load_existing_reports()
            combined_reports = historical_reports + cleaned_reports
            seen = set()
            deduped_combined = []
            for r in combined_reports:
                if r.id not in seen:
                    seen.add(r.id)
                    deduped_combined.append(r)
            combined_reports = deduped_combined
        else:
            combined_reports = cleaned_reports

        clustered_reports, incidents = self.clusterer.cluster(combined_reports)
        logger.info(f"-> Formed {len(incidents)} consolidated incidents across all active reports.")

        # STAGE 6: Prioritize incidents
        logger.info("\n--- [STAGE 6/7] Prioritizing Incidents (Urgency & Severity) ---")
        prioritized_incidents, final_reports = self.prioritizer.prioritize_incidents(incidents, clustered_reports)
        critical_count = sum(1 for inc in prioritized_incidents if inc.priority_level == "CRITICAL")
        high_count = sum(1 for inc in prioritized_incidents if inc.priority_level == "HIGH")
        medium_count = sum(1 for inc in prioritized_incidents if inc.priority_level == "MEDIUM")
        logger.info(f"-> Priority breakdown: {critical_count} CRITICAL, {high_count} HIGH, {medium_count} MEDIUM.")

        # =====================================================================
        # PART 3 & PART 4: Storing Data AFTER CLUSTERING & Final RESULTS
        # =====================================================================
        logger.info("\n--- [DATA PARTITIONS 3 & 4] Saving Data AFTER Clustering & Final Results ---")
        after_csv = self.exporter.export_after_clustering(prioritized_incidents)
        logger.info(f"-> Data After Clustering: {after_csv}")

        rep_csv, inc_csv = self.exporter.export_results(cleaned_reports, prioritized_incidents, append=incremental)
        logger.info(f"-> Final Results (Cleaned Reports): {rep_csv}")
        logger.info(f"-> Final Results (Incident Summaries): {inc_csv}")

        # Commit updated state checkpoint
        if incremental:
            self.state_tracker.update_state(cleaned_reports)

        elapsed = round(time.time() - start_time, 2)
        logger.info("=" * 70)
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed}s")
        logger.info("=" * 70)

        return {
            "status": "success",
            "elapsed_seconds": elapsed,
            "new_reports_count": len(cleaned_reports),
            "cleaned_reports_count": len(cleaned_reports),
            "total_reports_count": len(final_reports),
            "incidents_count": len(prioritized_incidents),
            "after_cleaning_csv": str(after_cleaning_csv),
            "before_clustering_csv": str(before_csv),
            "after_clustering_csv": str(after_csv),
            "results_reports_csv": str(rep_csv),
            "results_incidents_csv": str(inc_csv),
            "reports_csv_path": str(rep_csv),
            "incidents_csv_path": str(inc_csv),
            "priority_summary": {
                "critical": critical_count,
                "high": high_count,
                "medium": medium_count,
                "low": len(prioritized_incidents) - (critical_count + high_count + medium_count)
            },
            "incidents": prioritized_incidents,
            "reports": final_reports
        }
