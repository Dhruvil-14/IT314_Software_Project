"""
CLI Entry point for Disaster Coordination Hub (Gujarat).
Sprint 2 Pipeline Runner.
"""

import argparse
import sys
from pathlib import Path
from pipeline import DisasterHubPipeline

def main():
    parser = argparse.ArgumentParser(description="Disaster Coordination Hub (Gujarat) - Sprint 2 Pipeline")
    parser.add_argument("--mode", choices=["live", "offline", "auto"], default="auto",
                        help="Data collection mode: 'live' (internet feeds), 'offline' (sample test feeds), 'auto' (try live, fallback to offline)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Start completely fresh and overwrite previous CSV data (default: incremental append)")
    parser.add_argument("--download-images", action="store_true", default=True,
                        help="Download attached images locally for vision classifiers (default: True)")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip downloading binary images (faster execution)")
    parser.add_argument("--show-incidents", action="store_true", default=True,
                        help="Display prioritized incidents table in terminal output")
    args = parser.parse_args()

    use_live = (args.mode != "offline")
    fetch_images = not args.no_images
    is_incremental = not args.overwrite

    pipeline = DisasterHubPipeline()
    result = pipeline.run(
        use_live_feeds=use_live,
        download_images=fetch_images,
        incremental=is_incremental
    )

    if result.get("status") in ["success", "up_to_date"] and args.show_incidents:
        print("\n" + "=" * 90)
        print(" " * 28 + "PRIORITIZED GUJARAT INCIDENTS")
        print("=" * 90)
        print(f"{'INCIDENT ID':<22} | {'TYPE':<14} | {'DISTRICT':<15} | {'PRIORITY':<10} | {'SCORE':<6} | {'REPORTS':<7}")
        print("-" * 90)
        for inc in result.get("incidents", []):
            print(f"{inc.incident_id:<22} | {inc.disaster_type:<14} | {inc.primary_district:<15} | {inc.priority_level:<10} | {inc.urgency_score:<6.1f} | {inc.report_count:<7}")
        print("=" * 90)
        print(f"\n[OK] Pipeline Status: {result.get('status')}")
        print(f"  New Reports Added:      {result.get('new_reports_count', 0)}")
        print(f"  Total Active Reports:   {result.get('total_reports_count', 0)}")
        print(f"  Active Incidents:       {result.get('incidents_count', 0)}")
        print("\n" + "-" * 90)
        print("DATA DIRECTORY STRUCTURE (PARTITIONS):")
        print("-" * 90)
        print(f"  1. AFTER CLEANING:     {result['after_cleaning_csv']}")
        print(f"  2. BEFORE CLUSTERING:  {result['before_clustering_csv']}")
        print(f"  3. AFTER CLUSTERING:   {result['after_clustering_csv']}")
        print(f"  4. FINAL RESULTS:      {result['results_reports_csv']}")
        print(f"                         {result['results_incidents_csv']}")
        print(f"  5. MEDIA (IMAGES):     D:\\Study\\Python\\Project\\Data_collection_clustering\\data\\images\\\n")

if __name__ == "__main__":
    main()
