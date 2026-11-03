"""
Automated Test Suite for Disaster Coordination Hub (Gujarat).
Validates all 7 workflow stages of Sprint 2.
"""

import csv
import unittest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import DisasterHubPipeline
from cleaners.spam_filter import SpamFilter
from cleaners.gujarat_filter import GujaratFilter
from cleaners.deduplicator import Deduplicator
from collectors.models import Report
from extractors.information_extractor import InformationExtractor
from classifiers.text_classifier import DefaultTextClassifier
from classifiers.image_classifier import DefaultImageClassifier
from clustering.spatiotemporal_clusterer import SpatioTemporalClusterer, haversine_distance_km
from prioritizer.urgency_estimator import UrgencyEstimator
from config import OUTPUT_CLEANED_REPORTS, OUTPUT_INCIDENT_SUMMARIES

class TestDisasterPipeline(unittest.TestCase):

    def setUp(self):
        self.spam_filter = SpamFilter()
        self.gujarat_filter = GujaratFilter()
        self.deduplicator = Deduplicator()
        self.extractor = InformationExtractor()
        self.text_classifier = DefaultTextClassifier()
        self.image_classifier = DefaultImageClassifier()
        self.clusterer = SpatioTemporalClusterer()
        self.prioritizer = UrgencyEstimator()

    def test_01_spam_and_empty_filtering(self):
        """Verify empty posts, crypto spam, and non-disaster posts are caught."""
        # Empty post
        rep_empty = Report(
            id="t1", source="mstdn", source_type="social", title="..", text="hello",
            url="http://u", timestamp="", epoch_time=0.0, author="u"
        )
        valid, reason = self.spam_filter.evaluate_report(rep_empty)
        self.assertFalse(valid)
        self.assertIn("short", reason.lower())

        # Spam post
        rep_spam = Report(
            id="t2", source="mstdn", source_type="social", title="WIN BITCOIN",
            text="Storm of free crypto tokens for ahmedabad users! Claim bitcoin now at promo link.",
            url="http://u", timestamp="", epoch_time=0.0, author="u"
        )
        valid, reason = self.spam_filter.evaluate_report(rep_spam)
        self.assertFalse(valid)
        self.assertIn("spam", reason.lower())

        # Metaphorical post
        rep_meta = Report(
            id="t3", source="mstdn", source_type="social", title="Movie news",
            text="Flood of cinema fans gathered at theater in Ahmedabad city. The movie is on fire!",
            url="http://u", timestamp="", epoch_time=0.0, author="u"
        )
        valid, reason = self.spam_filter.evaluate_report(rep_meta)
        self.assertFalse(valid)
        self.assertIn("metaphorical", reason.lower())

        # Valid disaster post
        rep_valid = Report(
            id="t4", source="mstdn", source_type="social", title="Heavy flooding",
            text="Torrential rain caused severe waterlogging in Surat Varachha. Tapi river rising fast.",
            url="http://u", timestamp="", epoch_time=0.0, author="u"
        )
        valid, _ = self.spam_filter.evaluate_report(rep_valid)
        self.assertTrue(valid)

    def test_02_strict_gujarat_geofence(self):
        """Verify non-Gujarat states (Maharashtra, Kerala, etc.) are strictly rejected."""
        # Mumbai, Maharashtra post -> MUST BE REJECTED
        rep_mumbai = Report(
            id="g1", source="news", source_type="news", title="Mumbai Floods",
            text="Heavy waterlogging reported across Dadar and Hindmata in Mumbai Maharashtra. Train services halted.",
            url="http://u", timestamp="", epoch_time=0.0, author="u"
        )
        is_guj, reason, _, _, _ = self.gujarat_filter.verify_gujarat(rep_mumbai)
        self.assertFalse(is_guj)
        self.assertIn("out-of-state", reason.lower())

        # Surat, Gujarat post -> MUST BE ACCEPTED
        rep_surat = Report(
            id="g2", source="mstdn", source_type="social", title="Surat Tapi flood",
            text="Water overflow from Ukai dam entering low lying areas in Surat, Gujarat.",
            url="http://u", timestamp="", epoch_time=0.0, author="u"
        )
        is_guj, _, district, lat, lon = self.gujarat_filter.verify_gujarat(rep_surat)
        self.assertTrue(is_guj)
        self.assertEqual(district, "Surat")
        self.assertIsNotNone(lat)
        self.assertIsNotNone(lon)

        # Kutch district alias -> MUST BE ACCEPTED
        rep_kutch = Report(
            id="g3", source="news", source_type="news", title="Cyclone in Bhuj",
            text="Severe cyclonic winds hit Mandvi port and Bhuj in Kachchh district.",
            url="http://u", timestamp="", epoch_time=0.0, author="u"
        )
        is_guj, _, district, _, _ = self.gujarat_filter.verify_gujarat(rep_kutch)
        self.assertTrue(is_guj)
        self.assertEqual(district, "Kutch")

    def test_03_deduplication(self):
        """Verify duplicate reports are pruned."""
        r1 = Report(id="d1", source="mstdn", source_type="social", title="Flood in Surat",
                    text="Surat Varachha road is completely submerged in water. Tapi river rising.",
                    url="u1", timestamp="", epoch_time=100.0, author="a")
        r2 = Report(id="d2", source="mstdn", source_type="social", title="Flood in Surat",
                    text="Surat Varachha road is completely submerged in water. Tapi river rising.",
                    url="u2", timestamp="", epoch_time=105.0, author="b")
        unique, dups = self.deduplicator.deduplicate([r1, r2])
        self.assertEqual(len(unique), 1)
        self.assertEqual(dups, 1)

    def test_04_information_extraction(self):
        """Verify extraction of disaster type, place, coordinates, timestamp."""
        r = Report(
            id="ex1", source="news", source_type="news", title="Fire in Vatva GIDC",
            text="Massive chemical blaze erupted at industrial estate in Vatva GIDC Ahmedabad. Fire tenders rushed.",
            url="u", timestamp="2026-11-01T10:00:00Z", epoch_time=1793527200.0, author="press",
            district="Ahmedabad"
        )
        enriched = self.extractor.extract(r)
        self.assertEqual(enriched.disaster_type, "Fire")
        self.assertIn("Vatva", enriched.location_name)
        self.assertEqual(enriched.district, "Ahmedabad")
        self.assertIsNotNone(enriched.latitude)
        self.assertIsNotNone(enriched.longitude)

    def test_05_classifiers(self):
        """Verify text and image classifiers produce valid predictions and confidences."""
        text_res = self.text_classifier.predict("Severe cyclonic storm hitting coastal Kutch with 90 kmph winds")
        self.assertEqual(text_res.label, "Cyclone")
        self.assertGreater(text_res.confidence, 0.5)

        img_res = self.image_classifier.predict("https://example.com/disaster_flood_surat.jpg")
        self.assertEqual(img_res.label, "Flood")
        self.assertGreater(img_res.confidence, 0.7)

    def test_06_spatiotemporal_clustering(self):
        """Verify reports for Surat flood group into one incident, and Kutch cyclone into another."""
        r_surat1 = Report(id="s1", source="mstdn", source_type="social", title="Surat flood 1",
                          text="Surat flooded", url="", timestamp="", epoch_time=1000.0, author="a",
                          district="Surat", latitude=21.1702, longitude=72.8311, disaster_type="Flood")
        r_surat2 = Report(id="s2", source="news", source_type="news", title="Surat flood 2",
                          text="Surat Tapi flood", url="", timestamp="", epoch_time=1500.0, author="b",
                          district="Surat", latitude=21.1800, longitude=72.8350, disaster_type="Flood")
        r_kutch = Report(id="k1", source="news", source_type="news", title="Kutch cyclone",
                         text="Cyclone in Bhuj", url="", timestamp="", epoch_time=1200.0, author="c",
                         district="Kutch", latitude=23.2420, longitude=69.6669, disaster_type="Cyclone")

        _, incidents = self.clusterer.cluster([r_surat1, r_surat2, r_kutch])
        self.assertEqual(len(incidents), 2)
        surat_inc = next(i for i in incidents if i.primary_district == "Surat")
        self.assertEqual(surat_inc.report_count, 2)
        self.assertEqual(surat_inc.disaster_type, "Flood")

    def test_07_urgency_prioritization(self):
        """Verify official alerts and casualty indicators receive highest priority."""
        r_critical = Report(
            id="c1", source="gsdma", source_type="official_alert", title="OFFICIAL RED ALERT",
            text="GSDMA Official Warning: 5 casualties, dam breach imminent, evacuation ordered.",
            url="", timestamp="", epoch_time=1000.0, author="GSDMA", district="Surat",
            latitude=21.17, longitude=72.83, disaster_type="Flood", is_official_alert=True
        )
        _, incidents = self.clusterer.cluster([r_critical])
        prioritized, _ = self.prioritizer.prioritize_incidents(incidents, [r_critical])
        self.assertEqual(prioritized[0].priority_level, "CRITICAL")
        self.assertGreaterEqual(prioritized[0].urgency_score, 75.0)

    def test_08_end_to_end_pipeline(self):
        """Verify end-to-end execution and CSV generation."""
        pipeline = DisasterHubPipeline()
        result = pipeline.run(use_live_feeds=False, incremental=False)

        self.assertEqual(result["status"], "success")
        self.assertGreater(result["cleaned_reports_count"], 0)
        self.assertGreater(result["incidents_count"], 0)

        # Verify all partitioned CSV files exist on disk
        after_cleaning_csv = Path(result["after_cleaning_csv"])
        before_csv = Path(result["before_clustering_csv"])
        after_csv = Path(result["after_clustering_csv"])
        rep_csv = Path(result["results_reports_csv"])
        inc_csv = Path(result["results_incidents_csv"])

        self.assertTrue(after_cleaning_csv.exists(), "Part 1 after_cleaning CSV must exist")
        self.assertTrue(before_csv.exists(), "Part 2 before_clustering CSV must exist")
        self.assertTrue(after_csv.exists(), "Part 3 after_clustering CSV must exist")
        self.assertTrue(rep_csv.exists(), "Part 4 cleaned_reports CSV must exist")
        self.assertTrue(inc_csv.exists(), "Part 4 incident_summaries CSV must exist")

        # Verify after_cleaning CSV headers
        with open(after_cleaning_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            c_header = next(reader)
            self.assertIn("report_id", c_header)
            self.assertIn("cleaned_text", c_header)
            self.assertIn("is_gujarat_verified", c_header)

        # Verify Part 1 (before clustering) CSV headers
        with open(before_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            b_header = next(reader)
            self.assertIn("report_id", b_header)
            self.assertIn("disaster_type", b_header)
            self.assertIn("text_classifier_prediction", b_header)

        # Verify Part 2 (after clustering) CSV headers
        with open(after_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            a_header = next(reader)
            self.assertIn("incident_id", a_header)
            self.assertIn("centroid_lat", a_header)
            self.assertIn("associated_report_ids", a_header)

        # Verify Part 3 (final results) CSV headers
        with open(rep_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertIn("report_id", header)
            self.assertIn("district", header)
            self.assertIn("disaster_type", header)
            self.assertIn("cluster_incident_id", header)
            self.assertIn("urgency_score", header)

        with open(inc_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertIn("incident_id", header)
            self.assertIn("priority_level", header)
            self.assertIn("urgency_score", header)

if __name__ == "__main__":
    unittest.main()
