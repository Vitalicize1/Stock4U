from __future__ import annotations

from functools import lru_cache
from typing import Optional

import pandas as pd
import yfinance as yf


@lru_cache(maxsize=64)
def get_price_history_cached(ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    try:
        return yf.Ticker(ticker).history(period=period)
    except Exception:
        return None


