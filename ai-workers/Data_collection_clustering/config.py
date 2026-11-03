"""
Configuration and constants for Disaster Coordination Hub (Gujarat).
Sprint 2 Pipeline Configuration.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(r"D:/Study/Python/Project/Data_collection_clustering")
DATA_DIR = BASE_DIR / "data"

# Structured Data Partitions:
# 1. Data after cleaning (Step 2: deduplicated, spam-filtered, strict Gujarat verified)
DIR_AFTER_CLEANING = DATA_DIR / "after_cleaning"
OUTPUT_AFTER_CLEANING = DIR_AFTER_CLEANING / "reports_after_cleaning.csv"

# 2. Data before clustering (Step 4: extracted entities, coords & multi-modal classified)
DIR_BEFORE_CLUSTERING = DATA_DIR / "before_clustering"
OUTPUT_BEFORE_CLUSTERING = DIR_BEFORE_CLUSTERING / "reports_before_clustering.csv"

# 3. Data after clustering (Step 5: spatio-temporal grouped incident clusters)
DIR_AFTER_CLUSTERING = DATA_DIR / "after_clustering"
OUTPUT_AFTER_CLUSTERING = DIR_AFTER_CLUSTERING / "incidents_after_clustering.csv"

# 4. Final Results (Step 6 & 7: final cleaned reports with cluster IDs, urgency scores, actions)
DIR_RESULTS = DATA_DIR / "results"
OUTPUT_CLEANED_REPORTS = DIR_RESULTS / "cleaned_reports.csv"
OUTPUT_INCIDENT_SUMMARIES = DIR_RESULTS / "incident_summaries.csv"

# Images Directory
DIR_IMAGES = DATA_DIR / "images"

# Geographic Bounding Box for Gujarat State
GUJARAT_GEO_BOUNDS = {
    "min_lat": 20.05,
    "max_lat": 24.75,
    "min_lon": 68.10,
    "max_lon": 74.50,
}

# 33 Administrative Districts of Gujarat with Centroid Coordinates (WGS84)
GUJARAT_DISTRICTS = {
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "aliases": ["amdavad", "ahmedabad", "ahmadabad"]},
    "Amreli": {"lat": 21.6032, "lon": 71.2221, "aliases": ["amreli"]},
    "Anand": {"lat": 22.5645, "lon": 72.9289, "aliases": ["anand", "charotar"]},
    "Aravalli": {"lat": 23.5461, "lon": 73.1706, "aliases": ["aravalli", "aravali", "modasa"]},
    "Banaskantha": {"lat": 24.1724, "lon": 72.4346, "aliases": ["banaskantha", "banas kantha", "palanpur", "deesa"]},
    "Bharuch": {"lat": 21.7051, "lon": 72.9959, "aliases": ["bharuch", "broach", "ankleshwar", "dahej"]},
    "Bhavnagar": {"lat": 21.7645, "lon": 72.1519, "aliases": ["bhavnagar", "alang", "palitana"]},
    "Botad": {"lat": 22.1704, "lon": 71.6661, "aliases": ["botad", "salangpur", "gadhada"]},
    "Chhota Udepur": {"lat": 22.3108, "lon": 74.0135, "aliases": ["chhota udepur", "chhotaudaipur", "bodeli"]},
    "Dahod": {"lat": 22.8396, "lon": 74.2533, "aliases": ["dahod", "dohad"]},
    "Dang": {"lat": 20.8569, "lon": 73.6858, "aliases": ["dang", "dangs", "ahwa", "saputara"]},
    "Devbhumi Dwarka": {"lat": 22.2442, "lon": 68.9685, "aliases": ["devbhumi dwarka", "dwarka", "khambhalia", "okha"]},
    "Gandhinagar": {"lat": 23.2156, "lon": 72.6369, "aliases": ["gandhinagar", "kalol", "gift city"]},
    "Gir Somnath": {"lat": 20.9042, "lon": 70.3664, "aliases": ["gir somnath", "somnath", "veraval", "una", "kodinar"]},
    "Jamnagar": {"lat": 22.4707, "lon": 70.0577, "aliases": ["jamnagar", "sikka", "jodiya"]},
    "Junagadh": {"lat": 21.5222, "lon": 70.4579, "aliases": ["junagadh", "junagarh", "keshod", "mangrol"]},
    "Kheda": {"lat": 22.7533, "lon": 72.6844, "aliases": ["kheda", "nadiad", "kapadvanj", "matar"]},
    "Kutch": {"lat": 23.2420, "lon": 69.6669, "aliases": ["kutch", "kachchh", "bhuj", "gandhidham", "kandla", "mundra", "mandvi"]},
    "Mahisagar": {"lat": 23.1670, "lon": 73.5786, "aliases": ["mahisagar", "lunawada", "balasinor"]},
    "Mehsana": {"lat": 23.5880, "lon": 72.3693, "aliases": ["mehsana", "mahesana", "vadnagar", "kadi", "visnagar", "unjha"]},
    "Morbi": {"lat": 22.8120, "lon": 70.8377, "aliases": ["morbi", "wankaner", "maliya"]},
    "Narmada": {"lat": 21.8703, "lon": 73.5594, "aliases": ["narmada", "rajpipla", "kevadiya", "statue of unity"]},
    "Navsari": {"lat": 20.9500, "lon": 72.9300, "aliases": ["navsari", "bilimora", "chikhli", "vansda"]},
    "Panchmahal": {"lat": 22.7756, "lon": 73.6149, "aliases": ["panchmahal", "godhra", "halol", "champaner"]},
    "Patan": {"lat": 23.8493, "lon": 72.1266, "aliases": ["patan", "siddhpur", "radhanpur"]},
    "Porbandar": {"lat": 21.6417, "lon": 69.6293, "aliases": ["porbandar", "ranavav", "kutiyana"]},
    "Rajkot": {"lat": 22.3039, "lon": 70.8022, "aliases": ["rajkot", "gondal", "jetpur", "dhoraji"]},
    "Sabarkantha": {"lat": 23.6042, "lon": 72.9644, "aliases": ["sabarkantha", "himmatnagar", "idar", "khedbrahma"]},
    "Surat": {"lat": 21.1702, "lon": 72.8311, "aliases": ["surat", "bardoli", "kamrej", "hazira", "olpad"]},
    "Surendranagar": {"lat": 22.7284, "lon": 71.6370, "aliases": ["surendranagar", "wadhwan", "dhrangadhra", "limbdi", "chotila"]},
    "Tapi": {"lat": 21.2500, "lon": 73.4000, "aliases": ["tapi", "vyara", "songadh", "valod"]},
    "Vadodara": {"lat": 22.3072, "lon": 73.1812, "aliases": ["vadodara", "baroda", "padra", "waghodia", "savli"]},
    "Valsad": {"lat": 20.5992, "lon": 72.9342, "aliases": ["valsad", "vapi", "umbergaon", "parnera", "dharampur"]},
}

# Major River Basins and Key Infrastructure in Gujarat
GUJARAT_LANDMARKS = {
    "Sabarmati": {"lat": 23.0300, "lon": 72.5800, "district": "Ahmedabad"},
    "Narmada River": {"lat": 21.8400, "lon": 73.5400, "district": "Narmada"},
    "Tapi River": {"lat": 21.1800, "lon": 72.8000, "district": "Surat"},
    "Vishwamitri River": {"lat": 22.3100, "lon": 73.1900, "district": "Vadodara"},
    "Machchhu River": {"lat": 22.8200, "lon": 70.8400, "district": "Morbi"},
    "Sardar Sarovar Dam": {"lat": 21.8300, "lon": 73.7486, "district": "Narmada"},
    "Ukai Dam": {"lat": 21.2486, "lon": 73.5869, "district": "Tapi"},
    "Dharoi Dam": {"lat": 24.0042, "lon": 72.8531, "district": "Mehsana"},
    "Rann of Kutch": {"lat": 23.8000, "lon": 70.0000, "district": "Kutch"},
    "Saurashtra Coast": {"lat": 21.2000, "lon": 70.5000, "district": "Gir Somnath"},
}

# Other Indian states to blacklist if mentioned WITHOUT Gujarat context
OTHER_INDIAN_STATES = [
    "maharashtra", "mumbai", "pune", "kerala", "karnataka", "bengaluru", "tamil nadu",
    "chennai", "delhi", "punjab", "haryana", "uttar pradesh", "bihar", "odisha",
    "bhubaneswar", "west bengal", "kolkata", "rajasthan", "jaipur", "madhya pradesh",
    "bhopal", "indore", "andhra pradesh", "telangana", "hyderabad", "assam",
    "himachal pradesh", "uttarakhand", "jammu", "kashmir", "goa"
]

DIR_COLLECTED_DATA = DATA_DIR / "collected_data"

# Ensure directories exist at import time
for _dir in [DIR_AFTER_CLEANING, DIR_BEFORE_CLUSTERING, DIR_AFTER_CLUSTERING, DIR_RESULTS, DIR_IMAGES, DIR_COLLECTED_DATA]:
    _dir.mkdir(parents=True, exist_ok=True)
DISASTER_CATEGORIES = [
    "Flood",
    "Cyclone",
    "Earthquake",
    "Fire",
    "Landslide",
    "Heavy_Rain_Waterlogging",
    "Industrial_Hazard",
    "Heatwave",
    "Drought",
    "Structural_Collapse",
    "Other_Emergency"
]

# Priority Scoring Weights
PRIORITY_WEIGHTS = {
    "official_alert": 35.0,     # Weight for verified government alerts (NDMA/IMD/GSGMA)
    "severity_keywords": 30.0,  # Casualties, deaths, stranded, evacuation
    "evidence_count": 20.0,     # Multiple cross-source confirmations
    "recency": 10.0,            # Freshness of reports (within 6-12 hrs)
    "has_image": 5.0,           # Presence of visual media
}

# Clustering Parameters
SPATIAL_CLUSTER_RADIUS_KM = 30.0   # Reports within 30km clustered
TEMPORAL_CLUSTER_HOURS = 36.0       # Reports within 36 hours clustered

# Collector Endpoints
MASTODON_INSTANCES = [
    "https://mastodon.social",
    "https://mstdn.social",
    "https://mastodon.world",
]

MASTODON_TAGS = [
    "gujarat", "ahmedabad", "surat", "vadodara", "rajkot",
    "gujaratrain", "gujaratflood", "cyclone", "kutch", "weatherindia"
]

GDACS_RSS_URL = "https://www.gdacs.org/xml/rss.xml"
USGS_EARTHQUAKE_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
