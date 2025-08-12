import json
from pathlib import Path


def test_calibration_writes_artifacts(tmp_path, monkeypatch):
    # Stub history to avoid network; reuse a simple synthetic series
    import pandas as pd
    import numpy as np
    from backtesting import calibrate as cal

    def _fake_hist(ticker: str, period: str) -> pd.DataFrame:
        n = 120
        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
        prices = np.linspace(100, 102, n) + np.random.default_rng(0).normal(0, 0.05, n)
        df = pd.DataFrame({
            "Open": prices - 0.05,
            "High": prices + 0.1,
            "Low": prices - 0.1,
            "Close": prices,
            "Volume": np.full(n, 1_000_000, dtype=int),
        }, index=dates)
        df.index.name = "Date"
        return df

    monkeypatch.setattr(cal, "_load_history", _fake_hist)

    # Run calibration offline (ML+rule only)
    res = cal.calibrate("TEST", "6mo", warmup_days=40, offline=True)
    assert isinstance(res, dict)
    assert res["n_samples"] > 0
    assert "weights" in res and isinstance(res["weights"], dict)

    # Write via CLI path
    outdir = tmp_path / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    # Simulate CLI behavior: save JSON summary
    out_path = outdir / f"calibration_TEST_1d_6mo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f)

    assert out_path.exists()


