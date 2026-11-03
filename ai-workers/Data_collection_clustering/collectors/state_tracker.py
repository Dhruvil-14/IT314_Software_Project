"""
State Tracker and Watermarking for Disaster Coordination Hub.
Tracks last collection timestamp, seen report IDs, and content hashes
to support incremental runs without duplicating records or wiping previous data.
"""

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Set, Dict, Any, List, Optional
from collectors.models import Report

logger = logging.getLogger("DisasterHub.StateTracker")

class StateTracker:
    """Manages incremental run state and watermark checkpoints."""

    def __init__(self, state_file_path: Optional[Path] = None, csv_path: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.state_file = state_file_path or (base_dir / "data" / ".pipeline_state.json")
        self.csv_path = csv_path or (base_dir / "data" / "cleaned_reports.csv")

        self.last_run_epoch: float = 0.0
        self.last_run_timestamp: str = ""
        self.seen_report_ids: Set[str] = set()
        self.seen_hashes: Set[str] = set()
        self.total_lifetime_reports: int = 0

        self.load_state()

    def load_state(self):
        """Load state from checkpoint file and reconcile with existing CSV."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.last_run_epoch = data.get("last_run_epoch", 0.0)
                    self.last_run_timestamp = data.get("last_run_timestamp", "")
                    self.seen_report_ids = set(data.get("seen_report_ids", []))
                    self.seen_hashes = set(data.get("seen_hashes", []))
                    self.total_lifetime_reports = data.get("total_lifetime_reports", 0)
                logger.info(f"Loaded pipeline checkpoint: last run at {self.last_run_timestamp or 'N/A'}, {len(self.seen_report_ids)} known IDs.")
            except Exception as ex:
                logger.warning(f"Could not read state file: {ex}. Rebuilding from CSV.")

        # Reconcile with existing cleaned_reports.csv if IDs not populated
        if self.csv_path.exists() and len(self.seen_report_ids) == 0:
            try:
                with open(self.csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rid = row.get("report_id")
                        if rid:
                            self.seen_report_ids.add(rid)
                logger.info(f"Synchronized {len(self.seen_report_ids)} existing report IDs from {self.csv_path.name}.")
            except Exception as ex:
                logger.debug(f"CSV reconciliation error: {ex}")

    def filter_new_reports(self, reports: List[Report], filter_by_timestamp: bool = True) -> List[Report]:
        """
        Filter out reports that have already been collected in previous runs.
        Returns only genuinely new reports.
        """
        new_reports: List[Report] = []
        skipped_count = 0

        for r in reports:
            # Check 1: Already seen report ID
            if r.id in self.seen_report_ids:
                skipped_count += 1
                continue

            # Check 2: Timestamp filter if watermark exists
            if filter_by_timestamp and self.last_run_epoch > 0 and r.epoch_time > 0:
                # If report is older than the last run by more than 1 hour, ignore it
                if r.epoch_time < (self.last_run_epoch - 3600):
                    skipped_count += 1
                    continue

            new_reports.append(r)

        logger.info(f"Incremental Filter: Evaluated {len(reports)} raw reports -> {len(new_reports)} NEW, {skipped_count} ALREADY COLLECTED.")
        return new_reports

    def update_state(self, new_processed_reports: List[Report]):
        """Update and commit checkpoint after a successful pipeline run."""
        now = datetime.now(timezone.utc)
        self.last_run_timestamp = now.isoformat()
        self.last_run_epoch = now.timestamp()

        for r in new_processed_reports:
            self.seen_report_ids.add(r.id)

        self.total_lifetime_reports = len(self.seen_report_ids)

        # Write state atomically
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_run_timestamp": self.last_run_timestamp,
                "last_run_epoch": self.last_run_epoch,
                "seen_report_ids": list(self.seen_report_ids)[-5000:],  # keep last 5000 IDs
                "total_lifetime_reports": self.total_lifetime_reports,
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Updated pipeline checkpoint: total {self.total_lifetime_reports} tracked reports.")
        except Exception as ex:
            logger.warning(f"Failed to persist state checkpoint: {ex}")

    def reset_state(self):
        """Reset state tracking (e.g. when --overwrite is passed)."""
        self.last_run_epoch = 0.0
        self.last_run_timestamp = ""
        self.seen_report_ids.clear()
        self.seen_hashes.clear()
        self.total_lifetime_reports = 0
        if self.state_file.exists():
            try:
                self.state_file.unlink()
            except Exception:
                pass
        logger.info("Pipeline checkpoint state has been reset.")
