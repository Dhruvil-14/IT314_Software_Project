"""
Strict Gujarat Geographic Filter.
Ensures that collected reports are strictly within Gujarat and excludes other states.
"""

import re
from typing import Tuple, Optional
from collectors.models import Report
from config import GUJARAT_GEO_BOUNDS, GUJARAT_DISTRICTS, GUJARAT_LANDMARKS, OTHER_INDIAN_STATES

class GujaratFilter:
    """Strictly enforces Gujarat boundary, entity recognition, and out-of-state exclusion."""

    def __init__(self):
        self.bounds = GUJARAT_GEO_BOUNDS
        self.districts = GUJARAT_DISTRICTS
        self.landmarks = GUJARAT_LANDMARKS
        self.other_states = OTHER_INDIAN_STATES

    def verify_gujarat(self, report: Report) -> Tuple[bool, str, Optional[str], Optional[float], Optional[float]]:
        """
        Verify whether the report is exclusively for Gujarat.
        Returns:
            (is_gujarat, reason, identified_district, lat, lon)
        """
        # 1. Coordinate check if available
        if report.latitude is not None and report.longitude is not None:
            in_lat = self.bounds["min_lat"] <= report.latitude <= self.bounds["max_lat"]
            in_lon = self.bounds["min_lon"] <= report.longitude <= self.bounds["max_lon"]
            if not (in_lat and in_lon):
                return False, f"Coordinates ({report.latitude:.2f}, {report.longitude:.2f}) outside Gujarat bounds", None, None, None

        full_text = f"{report.title} {report.text}".lower()

        # 2. Check for out-of-state locations
        detected_other_states = [state for state in self.other_states if re.search(rf"\b{re.escape(state)}\b", full_text)]

        # 3. Match Gujarat districts, landmarks, and explicit mentions
        matched_district = None
        matched_lat = report.latitude
        matched_lon = report.longitude
        matched_entity = None

        # Check district names and aliases
        for dist_name, data in self.districts.items():
            for alias in data["aliases"]:
                if re.search(rf"\b{re.escape(alias)}\b", full_text):
                    matched_district = dist_name
                    matched_entity = alias
                    if matched_lat is None:
                        matched_lat = data["lat"]
                        matched_lon = data["lon"]
                    break
            if matched_district:
                break

        # Check landmarks if district not yet matched
        if not matched_district:
            for landmark_name, ldata in self.landmarks.items():
                if re.search(rf"\b{re.escape(landmark_name.lower())}\b", full_text):
                    matched_district = ldata["district"]
                    matched_entity = landmark_name
                    if matched_lat is None:
                        matched_lat = ldata["lat"]
                        matched_lon = ldata["lon"]
                    break

        has_general_gujarat = bool(re.search(r"\b(gujarat|gujarati|gsdma|saurashtra|kutch)\b", full_text))

        # Decision Logic:
        # If out-of-state is detected and NO Gujarat district or general mention exists -> Discard!
        if detected_other_states and not (matched_district or has_general_gujarat):
            return False, f"Out-of-state report detected ({', '.join(detected_other_states[:2])}) without Gujarat context", None, None, None

        # If neither district nor general Gujarat mention is present -> Discard!
        if not (matched_district or has_general_gujarat):
            return False, "No Gujarat location or district recognized", None, None, None

        # Default coordinates to state center if only general Gujarat was mentioned
        if matched_lat is None or matched_lon is None:
            if matched_district and matched_district in self.districts:
                matched_lat = self.districts[matched_district]["lat"]
                matched_lon = self.districts[matched_district]["lon"]
            else:
                # Centroid of Gujarat (Gandhinagar / Ahmedabad region)
                matched_lat = 23.0225
                matched_lon = 72.5714
                matched_district = "Gujarat (Statewide)"

        return True, f"Verified Gujarat ({matched_district})", matched_district, matched_lat, matched_lon
