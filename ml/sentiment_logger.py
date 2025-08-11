from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Optional


DATA_DIR = os.path.join("ml", "data")


def append_sentiment_sample(ticker: str, timestamp_iso: Optional[str], sentiment_score: float) -> None:
    """Append a daily sentiment sample to ml/data/sentiment_{ticker}.csv.

    - Ensures directory exists
    - Writes header if file doesn't exist
    - De-duplicates by date (YYYY-MM-DD): updates existing row for the same date
    """
    try:
        if not ticker:
            return
        os.makedirs(DATA_DIR, exist_ok=True)
        path = os.path.join(DATA_DIR, f"sentiment_{ticker.upper()}.csv")

        # Normalize date
        if timestamp_iso:
            try:
                dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
            except Exception:
                dt = datetime.utcnow()
        else:
            dt = datetime.utcnow()
        day_str = dt.strftime("%Y-%m-%d")

        rows = []
        if os.path.isfile(path):
            try:
                with open(path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        rows.append({"date": r.get("date", ""), "sentiment_score": r.get("sentiment_score", "")})
            except Exception:
                rows = []

        # Update or append
        updated = False
        for r in rows:
            if r.get("date") == day_str:
                r["sentiment_score"] = str(float(sentiment_score))
                updated = True
                break
        if not updated:
            rows.append({"date": day_str, "sentiment_score": str(float(sentiment_score))})

        # Write back
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "sentiment_score"])
            writer.writeheader()
            writer.writerows(rows)
    except Exception:
        # Silent failure to avoid disrupting main flow
        pass


