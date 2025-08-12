"""
CLI backtester that reuses PredictionAgent to simulate daily decisions.

Usage:
    python -m backtesting.run --ticker AAPL --period 1y [--cash 100000] [--fee_bps 5] [--slip_bps 5]

Notes:
    - Fetches daily OHLCV with yfinance via existing collector path
    - Rolls forward one bar at a time; at each step it builds a minimal state
      and calls PredictionAgent.make_prediction to get a direction and confidence
    - Simulates a simple long‑only position with configurable costs and slippage
    - Outputs a metrics summary (CAGR, Sharpe, max drawdown, hit rate)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import json

import numpy as np
import pandas as pd

from agents.prediction_agent import PredictionAgent
from portfolio.engine import PortfolioConfig, compute_target_weights, allowable_buy_notional
from agents.tools.data_collector_tools import collect_price_data
from backtesting.metrics import summarize_performance
from utils.result_cache import get_cached_result, set_cached_result


@dataclass
class BrokerConfig:
    starting_cash: float = 100_000.0
    fee_bps: float = 5.0  # round-trip assumed cost applied on trade notional (bps)
    slip_bps: float = 5.0  # slippage applied to fill price (bps)


def _apply_cost(price: float, fee_bps: float, slip_bps: float, side: int) -> float:
    """Return an effective fill price after costs.

    side: +1 for buy, -1 for sell.
    """
    slip = price * (slip_bps / 10_000.0) * side
    return price + slip


def _round_shares(cash: float, price: float) -> int:
    if price <= 0:
        return 0
    return int(cash // price)


def _load_history(ticker: str, period: str) -> pd.DataFrame:
    key = f"backtest_hist::{ticker}::{period}"
    cached = get_cached_result(key, ttl_seconds=24 * 3600)
    if cached and isinstance(cached, dict) and "hist" in cached:
        try:
            return pd.DataFrame(cached["hist"]).set_index("Date")
        except Exception:
            pass

    res = collect_price_data.invoke({"ticker": ticker, "period": period, "interval": "1d"})
    if res.get("status") != "success":
        raise RuntimeError(res.get("error", "failed to load history"))
    # Re-fetch with yfinance directly to get full OHLCV frame if needed
    # The tool already computed metrics; we just need the frame. Use yfinance through pandas_datareader pattern.
    # The tool does not return the raw history; load via yfinance again for full bars.
    import yfinance as yf
    hist = yf.Ticker(ticker).history(period=period, interval="1d")
    hist = hist.rename_axis("Date").reset_index().set_index("Date")

    set_cached_result(key, {"hist": hist.reset_index().to_dict(orient="list")})
    return hist


def _signal_policy_sma20(hist: pd.DataFrame, i: int) -> str:
    """SMA20 baseline: BUY if close > SMA20, SELL if close < SMA20, else HOLD."""
    win = 20
    if i < win:
        return "HOLD"
    sma20 = hist["Close"].rolling(window=win).mean()
    c = float(hist["Close"].iloc[i])
    s = float(sma20.iloc[i]) if not pd.isna(sma20.iloc[i]) else c
    if c > s:
        return "BUY"
    if c < s:
        return "SELL"
    return "HOLD"


def _signal_policy_rule(hist: pd.DataFrame, i: int) -> str:
    """Rule baseline: combine SMA20 with day momentum."""
    if i < 1:
        return "HOLD"
    sma_sig = _signal_policy_sma20(hist, i)
    up_day = float(hist["Close"].iloc[i]) > float(hist["Close"].iloc[i - 1])
    if sma_sig == "BUY" and up_day:
        return "BUY"
    if sma_sig == "SELL" and not up_day:
        return "SELL"
    return "HOLD"


def _simulate_portfolio(
    symbols: list[str],
    period: str,
    broker: BrokerConfig,
    offline: bool,
    use_ml_model: bool,
    outdir: str,
    warmup_days: int = 60,
) -> Dict[str, object]:
    # Load histories
    histories: Dict[str, pd.DataFrame] = {s: _load_history(s, period) for s in symbols}
    index = None
    for s, h in histories.items():
        if index is None or len(h.index) < len(index):
            index = h.index
    # Align by intersection of dates
    for s in symbols:
        histories[s] = histories[s].reindex(index)

    cash = broker.starting_cash
    positions: Dict[str, int] = {s: 0 for s in symbols}
    last_fill: Dict[str, float] = {s: None for s in symbols}
    equity_curve: List[float] = []
    daily_returns: List[float] = []
    trade_pnls: List[float] = []
    trades: List[Dict[str, object]] = []

    agent = PredictionAgent()
    config = PortfolioConfig()

    last_equity = cash
    for i in range(len(index)):
        # Mark-to-market
        equity = cash
        for s in symbols:
            price = float(histories[s]["Close"].iloc[i]) if not pd.isna(histories[s]["Close"].iloc[i]) else 0.0
            equity += positions[s] * price
        equity_curve.append(equity)
        if i > 0:
            daily_returns.append((equity - last_equity) / max(1e-9, last_equity))
        last_equity = equity

        if i < warmup_days:
            continue

        # Build inputs for engine
        per_ticker_inputs: Dict[str, Dict[str, Dict]] = {}
        prices: Dict[str, float] = {}
        for s in symbols:
            price = float(histories[s]["Close"].iloc[i])
            prices[s] = price
            state = {
                "ticker": s,
                "timeframe": "1d",
                "data": {"price_data": {"current_price": price, "previous_close": price}},
                "technical_analysis": {"technical_score": 50},
                "use_ml_model": use_ml_model,
            }
            if offline:
                state["low_api_mode"] = True
            pred = agent.make_prediction(state)
            per_ticker_inputs[s] = {
                "prediction_result": pred.get("prediction_result") or {},
                "confidence_metrics": (pred.get("final_prediction", {}) or {}).get("confidence_metrics", {})
            }

        target_w = compute_target_weights(per_ticker_inputs, config)

        # Compute allowable new buy notional for the day
        buy_bucket = allowable_buy_notional(cash, config)

        # Execute trades to move toward target weights
        for s in symbols:
            price = prices[s]
            if price <= 0:
                continue
            current_value = positions[s] * price
            target_value = target_w[s] * equity
            delta_value = target_value - current_value
            if abs(delta_value) < 1e-6:
                continue
            # Long-only handling: clamp negative delta to flat if shorts disabled
            if not config.allow_shorts and (current_value + delta_value) < 0:
                delta_value = -current_value

            if delta_value > 0:
                # Constrain by per-day buy bucket
                to_spend = min(delta_value, buy_bucket)
                if to_spend <= 0:
                    continue
                fill = _apply_cost(price, broker.fee_bps, broker.slip_bps, side=+1)
                qty = int(to_spend // max(fill, 1e-9))
                if qty <= 0:
                    continue
                spend = qty * fill
                cash -= spend
                buy_bucket -= spend
                positions[s] += qty
                last_fill[s] = fill
                trades.append({"date": str(index[i].date()), "symbol": s, "action": "BUY", "price": round(fill, 4), "shares": qty})
            else:
                # Sell
                fill = _apply_cost(price, broker.fee_bps, broker.slip_bps, side=-1)
                qty = min(abs(int(delta_value // max(fill, 1e-9))), positions[s])
                if qty <= 0:
                    continue
                proceeds = qty * fill
                cash += proceeds
                if last_fill[s] is not None:
                    trade_pnls.append((fill - last_fill[s]) * qty)
                positions[s] -= qty
                if positions[s] == 0:
                    last_fill[s] = None
                trades.append({"date": str(index[i].date()), "symbol": s, "action": "SELL", "price": round(fill, 4), "shares": qty})

    # Liquidate at end for reporting
    for s in symbols:
        price = float(histories[s]["Close"].iloc[-1])
        if positions[s] > 0:
            cash += positions[s] * price
            if last_fill[s] is not None:
                trade_pnls.append((price - last_fill[s]) * positions[s])
            trades.append({"date": str(index[-1].date()), "symbol": s, "action": "SELL", "price": round(price, 4), "shares": int(positions[s])})
            positions[s] = 0

    ending_equity = cash
    if equity_curve:
        equity_curve[-1] = ending_equity

    metrics = summarize_performance(equity_curve, daily_returns, trade_pnls)
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(trades).to_csv(out_path / f"trades_PORT_{period}.csv", index=False)
    except Exception:
        pass
    try:
        pd.DataFrame({"equity": equity_curve}).to_csv(out_path / f"equity_PORT_{period}.csv", index=False)
    except Exception:
        pass

    return {
        "symbols": symbols,
        "period": period,
        "starting_cash": broker.starting_cash,
        "ending_equity": ending_equity,
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "trade_pnls": trade_pnls,
        "trades": trades,
        "metrics": metrics,
    }


def simulate(
    ticker: str,
    period: str,
    broker: BrokerConfig,
    warmup_days: int = 60,
    use_ml_model: bool = False,
    offline: bool = True,
    policy: str = "agent",  # agent|rule|sma20
    outdir: Optional[str] = None,
) -> Dict[str, object]:
    hist = _load_history(ticker, period)
    if hist.empty:
        raise RuntimeError("empty history")

    prices = hist["Close"].astype(float)
    dates = prices.index

    agent = PredictionAgent()

    cash = broker.starting_cash
    shares = 0
    equity_curve: List[float] = []
    daily_returns: List[float] = []
    trade_pnls: List[float] = []
    trades: List[Dict[str, object]] = []

    last_equity = cash
    last_signal: Optional[str] = None
    last_fill_price: Optional[float] = None

    for i in range(len(prices)):
        price = float(prices.iloc[i])
        date = dates[i]

        # Update equity mark-to-market
        equity = cash + shares * price
        equity_curve.append(equity)
        if i > 0:
            daily_returns.append((equity - last_equity) / max(1e-9, last_equity))
        last_equity = equity

        if i < warmup_days:
            continue

        # Decide action by policy
        if policy == "agent":
            state = {
                "ticker": ticker,
                "timeframe": "1d",
                "data": {
                    "price_data": {
                        "current_price": price,
                        "previous_close": float(prices.iloc[i - 1]) if i > 0 else price,
                        "daily_change": price - (float(prices.iloc[i - 1]) if i > 0 else price),
                        "daily_change_pct": ((price / float(prices.iloc[i - 1]) - 1.0) * 100.0) if i > 0 else 0.0,
                        "volume": int(hist["Volume"].iloc[i]) if "Volume" in hist.columns else 0,
                        "high": float(hist["High"].iloc[i]),
                        "low": float(hist["Low"].iloc[i]),
                        "open": float(hist["Open"].iloc[i]),
                    },
                    "company_info": {},
                    "market_data": {},
                },
                "technical_analysis": {"technical_score": 50},
                "use_ml_model": use_ml_model,
            }
            if offline:
                state["low_api_mode"] = True
            pred = agent.make_prediction(state)
            pred_dict = pred.get("prediction_result") or pred
            direction = (pred_dict or {}).get("direction", "HOLD").upper()
            if direction in ("UP", "BUY", "STRONG_BUY"):
                action = "BUY"
            elif direction in ("DOWN", "SELL", "STRONG_SELL"):
                action = "SELL"
            else:
                action = "HOLD"
        elif policy == "sma20":
            action = _signal_policy_sma20(hist, i)
        elif policy == "rule":
            action = _signal_policy_rule(hist, i)
        else:
            action = "HOLD"

        # Execute at close with slippage
        if action == "BUY" and shares == 0:
            fill = _apply_cost(price, broker.fee_bps, broker.slip_bps, side=+1)
            qty = _round_shares(cash, fill)
            if qty > 0:
                cash -= qty * fill
                shares += qty
                last_fill_price = fill
                last_signal = "LONG"
                trades.append({"date": date.strftime('%Y-%m-%d'), "action": "BUY", "price": round(fill, 4), "shares": int(qty)})
        elif action == "SELL" and shares > 0:
            fill = _apply_cost(price, broker.fee_bps, broker.slip_bps, side=-1)
            cash += shares * fill
            if last_fill_price is not None:
                trade_pnls.append((fill - last_fill_price) * shares)
            trades.append({"date": date.strftime('%Y-%m-%d'), "action": "SELL", "price": round(fill, 4), "shares": int(shares)})
            shares = 0
            last_fill_price = None
            last_signal = "FLAT"

    # Liquidate at last price
    if shares > 0:
        final_fill = float(prices.iloc[-1])
        cash += shares * final_fill
        if last_fill_price is not None:
            trade_pnls.append((final_fill - last_fill_price) * shares)
        shares = 0

    final_equity = cash
    equity_curve[-1] = final_equity

    metrics = summarize_performance(equity_curve, daily_returns, trade_pnls)
    result = {
        "ticker": ticker,
        "period": period,
        "starting_cash": broker.starting_cash,
        "ending_equity": final_equity,
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "trade_pnls": trade_pnls,
        "trades": trades,
        "metrics": metrics,
        "policy": policy,
    }

    # Write artifacts
    if outdir:
        out_path = Path(outdir)
        out_path.mkdir(parents=True, exist_ok=True)
        base = f"{ticker}_{period}_{policy}"
        try:
            pd.DataFrame(trades).to_csv(out_path / f"trades_{base}.csv", index=False)
        except Exception:
            pass
        try:
            pd.DataFrame({"equity": equity_curve}).to_csv(out_path / f"equity_{base}.csv", index=False)
        except Exception:
            pass
        try:
            with open(out_path / f"backtest_{base}.json", "w", encoding="utf-8") as f:
                json.dump({"ticker": ticker, "period": period, "policy": policy, "metrics": metrics}, f, indent=2)
        except Exception:
            pass

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple backtest using PredictionAgent")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--ticker", help="Single ticker symbol, e.g., AAPL")
    g.add_argument("--tickers", help="Comma-separated list of symbols for portfolio backtest")
    parser.add_argument("--period", default="1y", help="Historical period (e.g., 6mo, 1y, 2y)")
    parser.add_argument("--cash", type=float, default=100_000.0, help="Starting cash")
    parser.add_argument("--fee_bps", type=float, default=5.0, help="Fee basis points per trade")
    parser.add_argument("--slip_bps", type=float, default=5.0, help="Slippage basis points")
    parser.add_argument("--ml", action="store_true", help="Use ML model path when available")
    parser.add_argument("--offline", action="store_true", help="Disable LLM calls and run offline")
    parser.add_argument("--policy", choices=["agent", "rule", "sma20"], default="agent", help="Decision policy")
    parser.add_argument("--outdir", default=os.path.join("cache", "results"), help="Directory to write artifacts")
    args = parser.parse_args()

    broker = BrokerConfig(starting_cash=args.cash, fee_bps=args.fee_bps, slip_bps=args.slip_bps)

    if args.tickers:
        # Multi-asset portfolio path
        symbols = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        result = _simulate_portfolio(
            symbols=symbols,
            period=args.period,
            broker=broker,
            offline=bool(args.offline),
            use_ml_model=args.ml,
            outdir=args.outdir,
        )
        m = result["metrics"]
        print("Portfolio Backtest Summary")
        print(f"Symbols: {','.join(symbols)}  Period: {args.period}")
        print(f"Start Cash: ${result['starting_cash']:.2f}  End Equity: ${result['ending_equity']:.2f}")
        print(f"Total Return: {m['total_return']*100:.2f}%  CAGR: {m['cagr']*100:.2f}%  Sharpe: {m['sharpe']:.2f}")
        print(f"Max Drawdown: {m['max_drawdown']*100:.2f}%  Hit Rate: {m['hit_rate']*100:.2f}%")
        print(f"Artifacts saved to: {args.outdir}")
        return

    # Single-asset path
    result = simulate(
            ticker=args.ticker.upper(),
            period=args.period,
            broker=broker,
            use_ml_model=args.ml,
            offline=bool(args.offline),
            policy=args.policy,
            outdir=args.outdir,
        )

    m = result["metrics"]
    print("Backtest Summary")
    print(f"Ticker: {result['ticker']}  Period: {result['period']}  Policy: {result['policy']}")
    print(f"Start Cash: ${result['starting_cash']:.2f}  End Equity: ${result['ending_equity']:.2f}")
    print(f"Total Return: {m['total_return']*100:.2f}%  CAGR: {m['cagr']*100:.2f}%  Sharpe: {m['sharpe']:.2f}")
    print(f"Max Drawdown: {m['max_drawdown']*100:.2f}%  Hit Rate: {m['hit_rate']*100:.2f}%")


if __name__ == "__main__":
    main()


