import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _force_offline_env(monkeypatch):
    """Force offline-friendly behavior for tests."""
    monkeypatch.setenv("DISABLE_LLM", "1")
    # Ensure cache dir exists on CI
    os.makedirs(os.path.join("cache", "results"), exist_ok=True)


@pytest.fixture(autouse=True)
def _mock_yfinance(monkeypatch):
    """Mock yfinance.Ticker.history to avoid network calls."""

    class FakeTicker:
        def __init__(self, *_args, **_kwargs):
            pass

        def history(self, period: str = "1mo", *args, **kwargs):
            # Generate a simple synthetic OHLCV for 60 business days
            num_days = 60
            end = datetime.now()
            start = end - timedelta(days=int(num_days * 1.6))
            idx = pd.bdate_range(start=start, end=end)
            base = 100.0
            # Random walk
            rng = np.random.default_rng(42)
            returns = rng.normal(0, 0.01, len(idx))
            prices = base * np.cumprod(1 + returns)
            df = pd.DataFrame(
                {
                    "Open": prices,
                    "High": prices * 1.01,
                    "Low": prices * 0.99,
                    "Close": prices,
                    "Volume": (rng.random(len(idx)) * 1_000_000).astype(int),
                },
                index=idx,
            )
            return df

    monkeypatch.setattr("yfinance.Ticker", FakeTicker, raising=True)


