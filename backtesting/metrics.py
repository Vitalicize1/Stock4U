"""
Backtesting metrics utilities.

Provides common portfolio/performance metrics for daily bar simulations.
All functions are side‑effect free and accept plain numpy/pandas inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


DEFAULT_TRADING_DAYS_PER_YEAR = 252


def compute_cagr(
    equity_curve: Iterable[float],
    periods_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
) -> float:
    """Compound annual growth rate from an equity curve.

    Args:
        equity_curve: Sequence of portfolio values over time.
        periods_per_year: Number of samples per year (252 for daily bars).

    Returns:
        CAGR as a float (e.g., 0.12 == 12%).
    """
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2 or np.any(eq <= 0):
        return 0.0
    n_periods = len(eq) - 1
    years = n_periods / float(periods_per_year)
    if years <= 0:
        return 0.0
    return float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0)


def compute_sharpe(
    returns: Iterable[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio using simple daily returns.

    Args:
        returns: Sequence of period returns (e.g., daily pct returns).
        risk_free_rate: Annual risk‑free rate in decimal form.
        periods_per_year: Number of samples per year.
    """
    r = np.asarray(list(returns), dtype=float)
    if r.size == 0:
        return 0.0

    # Convert annual risk‑free rate to period rate
    rf_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_period
    mu = np.nanmean(excess)
    sigma = np.nanstd(excess, ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float(np.sqrt(periods_per_year) * mu / sigma)


def compute_max_drawdown(equity_curve: Iterable[float]) -> Tuple[float, float]:
    """Max drawdown and drawdown duration.

    Returns:
        (max_drawdown, max_duration) where drawdown is in decimal (e.g., -0.25)
        and duration in number of periods.
    """
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size == 0:
        return 0.0, 0.0

    peaks = np.maximum.accumulate(eq)
    drawdowns = (eq - peaks) / peaks

    # Duration: consecutive periods under water
    duration = 0
    max_duration = 0
    for i in range(len(eq)):
        if eq[i] < peaks[i]:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0

    return float(np.min(drawdowns)), float(max_duration)


def compute_hit_rate(trade_pnls: Iterable[float]) -> float:
    """Share of trades with positive PnL.

    Args:
        trade_pnls: Sequence of PnL per closed trade.
    """
    vals = np.asarray(list(trade_pnls), dtype=float)
    if vals.size == 0:
        return 0.0
    wins = np.sum(vals > 0)
    return float(wins) / float(vals.size)


def summarize_performance(
    equity_curve: Iterable[float],
    daily_returns: Iterable[float],
    trade_pnls: Iterable[float],
    periods_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Compute a dictionary of common metrics from backtest series."""
    cagr = compute_cagr(equity_curve, periods_per_year)
    sharpe = compute_sharpe(daily_returns, risk_free_rate, periods_per_year)
    max_dd, dd_dur = compute_max_drawdown(equity_curve)
    hit = compute_hit_rate(trade_pnls)
    total_return = float(np.asarray(equity_curve, dtype=float)[-1] / float(np.asarray(equity_curve, dtype=float)[0]) - 1.0) if len(list(equity_curve)) >= 2 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_duration": dd_dur,
        "hit_rate": hit,
    }


