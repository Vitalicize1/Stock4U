"""
Ensemble calibration via backtests.

Runs a rolling, out-of-sample style pass over historical daily bars to:
- Collect component probabilities: ML, LLM (optional), Rule-based
- Grid-search weights to maximize AUC (classification on next-day up/down)
- Fit Platt scaling parameters (a, b) via logistic regression on the weighted
  probability to improve calibration
- Persist results into cache for live use by ensemble_prediction_tool

CLI:
  python -m backtesting.calibrate --ticker AAPL --period 1y [--include-llm]
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Tuple
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from utils.result_cache import set_cached_result
from agents.prediction_agent import PredictionAgent
from agents.tools.prediction_agent_tools import (
    generate_ml_prediction_tool,
    generate_llm_prediction_tool,
    generate_rule_based_prediction_tool,
    _direction_to_proba_up,
)


def _load_history(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    hist = yf.Ticker(ticker).history(period=period, interval="1d")
    if hist is None or hist.empty:
        raise RuntimeError(f"No history for {ticker} {period}")
    hist = hist.rename_axis("Date").reset_index().set_index("Date")
    return hist


def _build_state_for_index(ticker: str, hist: pd.DataFrame, i: int) -> Dict[str, object]:
    price = float(hist["Close"].iloc[i])
    prev = float(hist["Close"].iloc[i - 1]) if i > 0 else price
    return {
        "ticker": ticker,
        "timeframe": "1d",
        "data": {
            "price_data": {
                "current_price": price,
                "previous_close": prev,
                "daily_change": price - prev,
                "daily_change_pct": ((price / prev - 1.0) * 100.0) if i > 0 else 0.0,
                "volume": int(hist["Volume"].iloc[i]) if "Volume" in hist.columns else 0,
                "high": float(hist["High"].iloc[i]),
                "low": float(hist["Low"].iloc[i]),
                "open": float(hist["Open"].iloc[i]),
            },
            "company_info": {},
            "market_data": {},
        },
        "technical_analysis": {"technical_score": 50},
        "sentiment_analysis": {"overall_sentiment": {"sentiment_label": "neutral", "sentiment_score": 0.0}},
        "sentiment_integration": {"integrated_analysis": {"integrated_score": 50}},
    }


def _get_component_probs(
    state: Dict[str, object],
    analysis_summary: str,
    include_llm: bool,
    offline: bool,
) -> Dict[str, float]:
    probs: Dict[str, float] = {}

    # ML
    try:
        ml_res = generate_ml_prediction_tool.invoke({"state": state})
        ml_pred = (ml_res or {}).get("prediction_result")
        p = _direction_to_proba_up(ml_pred)
        if p is not None:
            probs["ml"] = float(p)
    except Exception:
        pass

    # LLM (optional)
    if include_llm and not offline:
        try:
            llm_res = generate_llm_prediction_tool.invoke({"analysis_summary": analysis_summary})
            llm_pred = (llm_res or {}).get("prediction_result")
            p = _direction_to_proba_up(llm_pred)
            if p is not None:
                probs["llm"] = float(p)
        except Exception:
            pass

    # Rule-based
    try:
        rule_res = generate_rule_based_prediction_tool.invoke({"analysis_summary": analysis_summary})
        rule_pred = (rule_res or {}).get("prediction_result")
        p = _direction_to_proba_up(rule_pred)
        if p is not None:
            probs["rule"] = float(p)
    except Exception:
        pass

    return probs


def _grid_search_weights(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    """Simple coarse grid search over weights to maximize AUC."""
    y = df["y"].values.astype(int)
    best_auc = -1.0
    best_w = None

    grid = np.linspace(0.0, 1.0, 6)  # 0.0,0.2,...,1.0
    for w_llm in (grid if "llm" in cols else [0.0]):
        for w_ml in (grid if "ml" in cols else [0.0]):
            for w_rule in (grid if "rule" in cols else [0.0]):
                w = {"llm": w_llm, "ml": w_ml, "rule": w_rule}
                # restrict to non-zero total
                s = sum(w[c] for c in cols)
                if s <= 0:
                    continue
                # normalize to available cols
                w = {k: (w[k] / s if k in cols else 0.0) for k in ["llm", "ml", "rule"]}
                p = np.zeros(len(df))
                for c in cols:
                    p += w[c] * df[c].values
                try:
                    auc = roc_auc_score(y, p)
                except Exception:
                    continue
                if auc > best_auc:
                    best_auc = auc
                    best_w = w
    # Fallback equal weights if something went wrong
    if best_w is None:
        eq = 1.0 / float(len(cols))
        best_w = {"llm": 0.0, "ml": 0.0, "rule": 0.0}
        for c in cols:
            best_w[c] = eq
    return best_w


def _fit_platt(p: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Fit logistic regression on probabilities p to calibrate.

    Returns (a, b, logloss)
    """
    X = p.reshape(-1, 1)
    clf = LogisticRegression(C=1e6, solver="lbfgs")
    clf.fit(X, y)
    a = float(clf.coef_[0][0])
    b = float(clf.intercept_[0])
    p_cal = clf.predict_proba(X)[:, 1]
    ll = log_loss(y, p_cal, labels=[0, 1])
    return a, b, ll


def calibrate(ticker: str, period: str, timeframe: str = "1d", warmup_days: int = 60, include_llm: bool = False, offline: bool = True) -> Dict[str, object]:
    hist = _load_history(ticker, period)
    if len(hist) <= warmup_days + 1:
        raise RuntimeError("Not enough data for calibration")

    pa = PredictionAgent()

    rows = []
    for i in range(warmup_days, len(hist) - 1):
        state = _build_state_for_index(ticker, hist, i)
        summary = pa._create_comprehensive_analysis_summary(
            ticker,
            state["data"]["price_data"],
            state.get("technical_analysis", {}),
            state.get("sentiment_analysis", {}),
            state.get("sentiment_integration", {}),
            state["data"].get("company_info", {}),
            state["data"].get("market_data", {}),
        )
        probs = _get_component_probs(state, summary, include_llm=include_llm, offline=offline)
        if not probs:
            continue
        # next-day binary label
        ret = float(hist["Close"].iloc[i + 1] / hist["Close"].iloc[i] - 1.0)
        y = 1 if ret > 0 else 0
        row = {**probs, "y": y}
        rows.append(row)

    df = pd.DataFrame(rows).dropna()
    if df.empty:
        raise RuntimeError("No component probabilities collected for calibration")

    cols = [c for c in ["llm", "ml", "rule"] if c in df.columns]
    weights = _grid_search_weights(df, cols)
    p_ens = np.zeros(len(df))
    for c in cols:
        p_ens += weights.get(c, 0.0) * df[c].values
    y = df["y"].values.astype(int)
    auc = roc_auc_score(y, p_ens)
    a, b, ll = _fit_platt(p_ens, y)

    # Persist to cache
    set_cached_result(f"ensemble_weights::{ticker.upper()}::{timeframe}", {
        "llm": float(weights.get("llm", 0.0)),
        "ml": float(weights.get("ml", 0.0)),
        "rule": float(weights.get("rule", 0.0)),
        "period": period,
        "n_samples": int(len(df)),
        "auc_raw": float(auc),
    })
    set_cached_result(f"ensemble_platt::{ticker.upper()}::{timeframe}", {
        "a": float(a),
        "b": float(b),
        "period": period,
        "n_samples": int(len(df)),
        "logloss": float(ll),
    })

    return {
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "period": period,
        "weights": weights,
        "platt": {"a": a, "b": b},
        "auc_raw": float(auc),
        "logloss_calibrated": float(ll),
        "n_samples": int(len(df)),
        "components": cols,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate ensemble weights and Platt parameters from backtests")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ticker", help="Single ticker to calibrate (e.g., AAPL)")
    g.add_argument("--tickers", help="Comma-separated list of tickers (e.g., AAPL,MSFT,NVDA)")
    p.add_argument("--period", default="1y")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--include-llm", action="store_true", help="Attempt to include LLM component if keys available")
    p.add_argument("--offline", action="store_true", help="Disable LLM calls and run offline")
    p.add_argument("--warmup", type=int, default=60, help="Warmup days before first sample")
    p.add_argument("--outdir", default=os.path.join("cache", "results"), help="Directory to write calibration summaries as JSON")
    p.add_argument("--show", action="store_true", help="Print cached weights and Platt after saving")
    args = p.parse_args()

    # Build ticker list
    tickers: List[str] = []
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.ticker:
        tickers = [args.ticker.strip().upper()]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for t in tickers:
        res = calibrate(
            t,
            args.period,
            timeframe=args.timeframe,
            warmup_days=args.warmup,
            include_llm=args.include_llm,
            offline=bool(args.offline) or (not args.include_llm),
        )

        # Save JSON artifact
        out_path = outdir / f"calibration_{t}_{args.timeframe}_{args.period}.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        except Exception:
            pass

        print("Calibration Summary")
        print(f"Ticker={res['ticker']} Timeframe={res['timeframe']} Period={res['period']} Samples={res['n_samples']}")
        print(f"Weights: {res['weights']}")
        print(f"Platt: a={res['platt']['a']:.4f}, b={res['platt']['b']:.4f}")
        print(f"AUC(raw)={res['auc_raw']:.4f}  LogLoss(calibrated)={res['logloss_calibrated']:.4f}")

        if args.show:
            from utils.result_cache import get_cached_result
            kw = get_cached_result(f"ensemble_weights::{t}::{args.timeframe}", ttl_seconds=10**9)
            kp = get_cached_result(f"ensemble_platt::{t}::{args.timeframe}", ttl_seconds=10**9)
            print("Cached Weights:", kw)
            print("Cached Platt:", kp)


if __name__ == "__main__":
    main()


