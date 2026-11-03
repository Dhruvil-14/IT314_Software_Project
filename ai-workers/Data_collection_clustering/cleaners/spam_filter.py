"""
Spam and Irrelevance Filter for Disaster Hub.
Filters out commercial spam, clickbait, trivial empty posts, and metaphorical/non-disaster contexts.
"""

import re
from typing import Tuple
from collectors.models import Report

# Commercial and spam terms
SPAM_PATTERNS = [
    r"\bcrypto\b", r"\bbitcoin\b", r"\bforex\b", r"\bcasino\b", r"\bgiveaway\b",
    r"\bfree tokens\b", r"\bdiscount\b", r"\bsale\b", r"\bstream free\b",
    r"\bclick here\b", r"\bwin \$\d+", r"\bpromo code\b", r"\btelegram channel\b",
    r"\bfollow back\b", r"\bsubscribe now\b", r"\bweight loss\b", r"\bmake money\b"
]

# Non-disaster metaphorical phrases
METAPHOR_PATTERNS = [
    r"flood of (\w+\s+)?(offers|comments|complaints|fans|people|queries|visitors|requests|emails|traffic|customers)",
    r"(market|stocks|shares|economy) (crash|earthquake)",
    r"(box office|actor|dance|performance|song|theater) (is |was )?on fire",
    r"political (storm|cyclone|tempest)",
    r"fire sales?",
    r"fired from (job|company|work)"
]

# Essential disaster vocabulary - must contain at least one disaster signal to be relevant
DISASTER_VOCABULARY = [
    r"\bflood(s|ing|ed)?\b", r"\binundat(ion|ed)\b", r"\bwaterlog(ging|ged)?\b",
    r"\bcyclon(e|ic)\b", r"\bhurrican(e)?\b", r"\btyphoon\b", r"\bheavy rain(fall|s)?\b",
    r"\btorrential\b", r"\bdownpour\b", r"\bearthquak(e|es)\b", r"\btremor(s)?\b",
    r"\bseismic\b", r"\bblaz(e|es|ing)\b", r"\bfire(s)?\b", r"\blandslid(e|es)\b",
    r"\bmudslid(e|es)\b", r"\bheatwave\b", r"\bdrought\b", r"\bcollapse(d|s)?\b",
    r"\bstran(ded|ding)\b", r"\bevacuat(e|ed|ion)\b", r"\brescue(d|s|r)?\b",
    r"\bcasualt(y|ies)\b", r"\bdeath toll\b", r"\bsubmerg(ed|e)\b", r"\bndrf\b",
    r"\bsdrf\b", r"\bgsdma\b", r"\bdisaster(s)?\b", r"\bemergency\b", r"\balert(s)?\b",
    r"\bwarn(ing|ed)\b", r"\bred alert\b", r"\borange alert\b", r"\byellow alert\b",
    r"\bdam breach\b", r"\boverflow(ing)?\b", r"\bcusecs\b"
]

class SpamFilter:
    """Detects and filters spam, empty posts, and non-disaster irrelevant texts."""

    def __init__(self, min_char_length: int = 25):
        self.min_char_length = min_char_length

    def evaluate_report(self, report: Report) -> Tuple[bool, str]:
        """
        Evaluate if a report is valid, spam, empty, or irrelevant.
        Returns: (is_valid, reason)
        """
        combined_text = f"{report.title} {report.text}".strip()

        # 1. Empty or too short check
        if len(combined_text) < self.min_char_length:
            return False, f"Empty or too short (<{self.min_char_length} chars)"

        # Check if text contains sufficient alphabetic characters
        letters = re.findall(r"[a-zA-Z]", combined_text)
        if len(letters) < 15:
            return False, "Insufficient alphabetic content"

        lower_text = combined_text.lower()

        # 2. Commercial / spam keyword check
        for pattern in SPAM_PATTERNS:
            if re.search(pattern, lower_text):
                return False, f"Spam pattern matched: {pattern}"

        # 3. Metaphorical false positive check
        for pattern in METAPHOR_PATTERNS:
            if re.search(pattern, lower_text):
                return False, f"Metaphorical/non-disaster match: {pattern}"

        # 4. Disaster domain relevance check
        # Official alerts bypass strict keyword check if already marked official
        if report.is_official_alert:
            return True, "Valid official alert"

        has_disaster_cue = any(re.search(pat, lower_text) for pat in DISASTER_VOCABULARY)
        if not has_disaster_cue:
            return False, "No disaster-relevant vocabulary found"

        return True, "Valid disaster report"
