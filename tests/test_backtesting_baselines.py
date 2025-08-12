import os
from pathlib import Path
import pandas as pd
import numpy as np


def _make_hist(n_days: int = 120) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="B")
    # Upward trend with noise
    prices = np.linspace(100, 120, n_days) + np.random.default_rng(42).normal(0, 0.2, n_days)
    open_p = prices - 0.1
    high = prices + 0.3
    low = prices - 0.3
    vol = np.full(n_days, 1_000_000, dtype=int)
    df = pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": prices,
        "Volume": vol,
    }, index=dates)
    df.index.name = "Date"
    return df


def test_backtesting_baseline_policies(tmp_path, monkeypatch):
    from backtesting import run as bt

    # Avoid network by stubbing history loader
    monkeypatch.setattr(bt, "_load_history", lambda ticker, period: _make_hist(80))

    outdir = tmp_path / "results"
    outdir_str = str(outdir)

    # Run SMA20 baseline
    res_sma = bt.simulate(
        ticker="TEST",
        period="6mo",
        broker=bt.BrokerConfig(starting_cash=10_000.0, fee_bps=0.0, slip_bps=0.0),
        warmup_days=25,
        offline=True,
        policy="sma20",
        outdir=outdir_str,
    )

    assert res_sma["metrics"].get("sharpe") is not None
    assert isinstance(res_sma.get("trades"), list)

    # Run rule baseline
    res_rule = bt.simulate(
        ticker="TEST",
        period="6mo",
        broker=bt.BrokerConfig(starting_cash=10_000.0, fee_bps=0.0, slip_bps=0.0),
        warmup_days=25,
        offline=True,
        policy="rule",
        outdir=outdir_str,
    )
    assert res_rule["metrics"].get("cagr") is not None

    # Artifacts should exist
    trades_csv = outdir / "trades_TEST_6mo_sma20.csv"
    equity_csv = outdir / "equity_TEST_6mo_sma20.csv"
    summary_json = outdir / "backtest_TEST_6mo_sma20.json"
    assert trades_csv.exists()
    assert equity_csv.exists()
    assert summary_json.exists()


