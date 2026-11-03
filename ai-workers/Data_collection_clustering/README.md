# Gujarat Disaster Coordination Hub - Sprint 2 Pipeline

A real-time data ingestion, cleaning, normalization, information extraction, multi-modal classification, spatio-temporal clustering, and urgency prioritization pipeline strictly scoped to **Gujarat, India**.

---

## 📁 Data Directory Structure (Partitioned Lifecycle)

The `data/` directory organizes the data across each stage of the Sprint 2 pipeline:

```
data/
├── 1. after_cleaning/
│   └── reports_after_cleaning.csv        <-- PART 1: Raw data immediately AFTER cleaning & normalization (Step 2)
│
├── 2. before_clustering/
│   └── reports_before_clustering.csv     <-- PART 2: Cleaned data AFTER extraction & classification, BEFORE clustering (Step 4)
│
├── 3. after_clustering/
│   └── incidents_after_clustering.csv    <-- PART 3: Grouped incident clusters AFTER spatio-temporal clustering (Step 5)
│
├── 4. results/
│   ├── cleaned_reports.csv               <-- PART 4: Final cleaned reports with cluster IDs & urgency scores
│   └── incident_summaries.csv            <-- PART 4: Actionable incident summaries & response recommendations
│
└── images/                               <-- MEDIA: Downloaded local image files (.jpg, .png, etc.)
```

---

### Detailed Partition Contents

#### 📌 1. `data/after_cleaning/reports_after_cleaning.csv`
Output of **Step 2 (Clean and Normalize)**:
- Filtered for empty posts, commercial spam, and non-disaster metaphors.
- Strict Gujarat geo-fencing (out-of-state records like Mumbai/Kerala removed).
- Exact and near-duplicate reports pruned.
- Fields: `report_id`, `source`, `source_type`, `timestamp`, `district`, `location_name`, `latitude`, `longitude`, `is_official`, `title`, `cleaned_text`, `url`, `media_count`, `is_gujarat_verified`.

#### 📌 2. `data/before_clustering/reports_before_clustering.csv`
Output of **Step 3 (Extraction) & Step 4 (Classification)**:
- Resolved local landmarks, talukas, and standardized timestamps.
- Predictions from the text and image classifiers with confidence probabilities.
- Fields: `report_id`, `source`, `source_type`, `timestamp`, `district`, `location_name`, `latitude`, `longitude`, `disaster_type`, `text_classifier_prediction`, `text_confidence`, `image_classifier_prediction`, `image_confidence`, `is_official`, `title`, `text_snippet`, `url`, `media_count`, `local_images`.

#### 📌 3. `data/after_clustering/incidents_after_clustering.csv`
Output of **Step 5 (Cluster Related Reports)**:
- Reports describing the same incident at the same location grouped together using Haversine distance and temporal windowing.
- Fields: `incident_id`, `disaster_type`, `district`, `location_name`, `centroid_lat`, `centroid_lon`, `first_reported`, `last_reported`, `report_count`, `social_reports`, `news_reports`, `official_alerts`, `has_official_alert`, `urgency_score`, `priority_level`, `recommended_actions`, `associated_report_ids`, `key_evidence_summary`.

#### 📌 4. `data/results/`
Final operational deliverables:
- **`cleaned_reports.csv`**: Every cleaned individual report mapped to its assigned `cluster_incident_id`, `urgency_score`, `severity_level`, and downloaded image paths.
- **`incident_summaries.csv`**: Executive situational dashboard with prioritized incidents and operational recommendations.

---

## 🚀 How to Run

From PowerShell in `D:\Study\Python\Project\Data_collection_clustring`:

### 1. Incremental Live Mode (Appends New Live Data)
```powershell
python main.py --mode live --show-incidents
```

### 2. Fresh Run (Reset and Overwrite All Partitions)
```powershell
python main.py --mode live --overwrite --show-incidents
```

### 3. Run Pipeline Without Downloading Images
```powershell
python main.py --mode live --no-images --show-incidents
```

### 4. Run Automated Test Suite
```powershell
python -m unittest tests/test_pipeline.py
```
*(All 8 test cases validating each stage and all data partitions pass with 100% success).*
