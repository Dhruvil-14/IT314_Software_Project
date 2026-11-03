"""
Information Extraction Engine for Step 3.
Extracts disaster type, specific local place names, standardized timestamps, and coordinates.
"""

import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from collectors.models import Report
from config import GUJARAT_DISTRICTS, GUJARAT_LANDMARKS

# Keyword patterns mapping to disaster types
DISASTER_TYPE_RULES = {
    "Flood": [
        r"\bflood(s|ing|ed)?\b", r"\binundat(ion|ed)\b", r"\bwaterlog(ging|ged)?\b",
        r"\bsubmerg(ed|ing)\b", r"\bwater release\b", r"\bdam overflow\b", r"\brising river\b"
    ],
    "Cyclone": [
        r"\bcyclon(e|ic)\b", r"\bhurrican(e)?\b", r"\bgale(s)?\b", r"\bsquall(y)?\b",
        r"\bstorm surge\b", r"\bwind gusts?\b", r"\btidal wave\b", r"\bdeep depression\b"
    ],
    "Earthquake": [
        r"\bearthquak(e|es)\b", r"\btremor(s)?\b", r"\bseismic\b", r"\brichter\b",
        r"\bmagnitude\b", r"\bepicenter\b"
    ],
    "Fire": [
        r"\bblaz(e|ing|es)\b", r"\bfire\b", r"\binferno\b", r"\bflames?\b",
        r"\bchemical explosion\b", r"\bfire tender(s)?\b", r"\bfire brigade\b"
    ],
    "Structural_Collapse": [
        r"\bcollaps(e|ed|ing)\b", r"\bdebris\b", r"\brubble\b", r"\bbuilding fall(en)?\b",
        r"\btrapped under debris\b"
    ],
    "Landslide": [
        r"\blandslid(e|es)\b", r"\bmudslid(e|es)\b", r"\brockfall\b"
    ],
    "Heatwave": [
        r"\bheatwave(s)?\b", r"\bextreme heat\b", r"\bsunstroke\b"
    ],
    "Drought": [
        r"\bdrought\b", r"\bwater scarcity\b", r"\bdry spell\b"
    ]
}

# Talukas, local areas and neighborhoods across Gujarat for fine-grained place extraction
LOCAL_AREAS = {
    "Varachha": "Surat", "Rander": "Surat", "Adajan": "Surat", "Hazira": "Surat",
    "Katargam": "Surat", "Udhna": "Surat", "Olpad": "Surat",
    "Mandvi": "Kutch", "Mundra": "Kutch", "Bhuj": "Kutch", "Gandhidham": "Kutch",
    "Kandla": "Kutch", "Anjar": "Kutch",
    "Vatva": "Ahmedabad", "Narol": "Ahmedabad", "Maninagar": "Ahmedabad",
    "Bopal": "Ahmedabad", "Sanand": "Ahmedabad", "Viramgam": "Ahmedabad",
    "Fatehgunj": "Vadodara", "Sayajiganj": "Vadodara", "Kala Ghoda": "Vadodara",
    "Manjalpur": "Vadodara", "Makarpura": "Vadodara",
    "Dhebar Road": "Rajkot", "Kalawad Road": "Rajkot", "150 Feet Ring Road": "Rajkot",
    "Veraval": "Gir Somnath", "Somnath": "Gir Somnath", "Una": "Gir Somnath",
    "Dwarka": "Devbhumi Dwarka", "Khambhalia": "Devbhumi Dwarka",
    "Alang": "Bhavnagar", "Palitana": "Bhavnagar",
    "Palanpur": "Banaskantha", "Deesa": "Banaskantha",
    "Nadiad": "Kheda", "Godhra": "Panchmahal", "Halol": "Panchmahal",
    "Vapi": "Valsad", "Umbergaon": "Valsad", "Bilimora": "Navsari",
    "Morbi Ceramic Zone": "Morbi", "Wankaner": "Morbi"
}

class InformationExtractor:
    """Extracts disaster types, place names, standardized timestamps, and coordinates."""

    def extract(self, report: Report) -> Report:
        """Enrich report with extracted features."""
        combined_text = f"{report.title} {report.text}"

        # 1. Extract Disaster Type
        disaster_type = self._extract_disaster_type(combined_text, default=report.disaster_type)
        report.disaster_type = disaster_type

        # 2. Extract Place Names (Specific Local Area + District)
        place_name, district = self._extract_place_names(combined_text, current_district=report.district)
        if place_name:
            report.location_name = place_name
        if district and not report.district:
            report.district = district

        # 3. Extract / Standardize Coordinates
        lat, lon = self._resolve_coordinates(report.latitude, report.longitude, report.district, report.location_name)
        report.latitude = lat
        report.longitude = lon

        # 4. Standardize Timestamp
        iso_str, epoch = self._standardize_timestamp(report.timestamp, report.epoch_time)
        report.timestamp = iso_str
        report.epoch_time = epoch

        return report

    def _extract_disaster_type(self, text: str, default: str = "Unclassified") -> str:
        """Match textual patterns to standard disaster categories."""
        text_lower = text.lower()
        matched_scores = {}

        for dtype, patterns in DISASTER_TYPE_RULES.items():
            score = 0
            for pat in patterns:
                matches = len(re.findall(pat, text_lower))
                score += matches
            if score > 0:
                matched_scores[dtype] = score

        if matched_scores:
            # Return disaster type with highest occurrences
            return max(matched_scores, key=matched_scores.get)
        
        return default if default != "Unclassified" else "Other_Emergency"

    def _extract_place_names(self, text: str, current_district: Optional[str]) -> Tuple[str, Optional[str]]:
        """Identify specific local neighborhood/taluka and parent district."""
        matched_local = None
        matched_district = current_district

        # Check local neighborhoods/talukas
        for area, dist in LOCAL_AREAS.items():
            if re.search(rf"\b{re.escape(area)}\b", text, re.IGNORECASE):
                matched_local = area
                if not matched_district or matched_district == "Gujarat (Statewide)":
                    matched_district = dist
                break

        # Check landmarks
        if not matched_local:
            for lmark, ldata in GUJARAT_LANDMARKS.items():
                if re.search(rf"\b{re.escape(lmark)}\b", text, re.IGNORECASE):
                    matched_local = lmark
                    if not matched_district:
                        matched_district = ldata["district"]
                    break

        if matched_local and matched_district:
            full_place = f"{matched_local}, {matched_district}"
        elif matched_district:
            full_place = matched_district
        elif matched_local:
            full_place = matched_local
        else:
            full_place = "Gujarat"

        return full_place, matched_district

    def _resolve_coordinates(self, lat: Optional[float], lon: Optional[float],
                             district: Optional[str], location_name: Optional[str]) -> Tuple[float, float]:
        """Ensure valid coordinates exist for mapping and spatial clustering."""
        if lat is not None and lon is not None:
            return lat, lon

        # Check if local area in landmarks
        if location_name:
            for lmark, ldata in GUJARAT_LANDMARKS.items():
                if lmark.lower() in location_name.lower():
                    return ldata["lat"], ldata["lon"]

        # Check district gazetteer
        if district and district in GUJARAT_DISTRICTS:
            return GUJARAT_DISTRICTS[district]["lat"], GUJARAT_DISTRICTS[district]["lon"]

        # Default fallback to Gandhinagar / Ahmedabad state center
        return 23.0225, 72.5714

    def _standardize_timestamp(self, ts_str: str, epoch: float) -> Tuple[str, float]:
        """Guarantee valid ISO 8601 UTC string and unix epoch timestamp."""
        if epoch and epoch > 0:
            try:
                dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
                return dt.isoformat(), epoch
            except Exception:
                pass

        if ts_str:
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                return dt.isoformat(), dt.timestamp()
            except Exception:
                pass

        now = datetime.now(timezone.utc)
        return now.isoformat(), now.timestamp()
