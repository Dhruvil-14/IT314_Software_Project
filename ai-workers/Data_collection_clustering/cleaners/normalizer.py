"""
Text Normalization for Disaster Hub Reports.
Cleans raw text, handles character encoding, strips markup, and normalizes whitespaces.
"""

import html
import re

def clean_text(text: str) -> str:
    """Normalize and clean raw textual content."""
    if not text:
        return ""
    
    # Unescape HTML entities (&amp;, &lt;, etc.)
    cleaned = html.unescape(text)
    
    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    
    # Normalize URLs into space or placeholder if inside text
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    
    # Remove control characters and excessive punctuation
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"[^\w\s\.,;:!?\-'\"]", " ", cleaned)
    
    # Collapse multiple spaces into single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
