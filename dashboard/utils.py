from __future__ import annotations

import re
from typing import Any


def humanize_label(value: Any, default: str = "Unknown") -> str:
    """Convert enum-like labels into human-friendly text for UI only.

    Examples:
    - "STRONG_BUY" -> "Strong Buy"
    - "BUY_WEAK" -> "Buy Weak"
    - "very_positive" -> "Very Positive"
    - "UP" -> "Up"
    - None -> default
    """
    try:
        if value is None:
            return default
        text = str(value)
        # Replace common separators with spaces
        text = text.replace("_", " ").replace("-", " ")
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Title case for readability
        return text.title()
    except Exception:
        return default


