from __future__ import annotations

import os
from typing import Optional
import pandas as pd


def read_sentiment_csv(ticker: str, index: pd.Index, data_dir: str = os.path.join("ml", "data")) -> Optional[pd.Series]:
    """Read ml/data/sentiment_{ticker}.csv with columns [date, sentiment_score] and align to index.

    Returns a Series indexed like `index` or None if file not found/invalid.
    """
    try:
        path = os.path.join(data_dir, f"sentiment_{ticker.upper()}.csv")
        if not os.path.isfile(path):
            return None
        df = pd.read_csv(path)
        # Find date column
        date_col = None
        for c in df.columns:
            if str(c).lower() in ("date", "ds", "timestamp"):
                date_col = c
                break
        score_col = None
        for c in df.columns:
            if str(c).lower() in ("sentiment_score", "score", "overall_sentiment", "sentiment"):
                score_col = c
                break
        if date_col is None or score_col is None:
            return None
        df[date_col] = pd.to_datetime(df[date_col])
        df = df[[date_col, score_col]].dropna()
        df = df.set_index(date_col).sort_index()
        # Align to index with nearest match within tolerance
        aligned = df[score_col].reindex(index, method="nearest", tolerance=pd.Timedelta(days=2))
        return aligned
    except Exception:
        return None


