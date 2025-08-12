from __future__ import annotations

"""
Accuracy Baseline runner

Runs a small suite of backtests across fixed tickers/timeframes and
produces a concise metrics report for agent vs. simple baselines.

Artifacts are written to cache/metrics/accuracy_baseline/.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import time
import contextlib
import io

import pandas as pd

from backtesting.run import simulate, BrokerConfig


DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
DEFAULT_PERIOD = "1y"
DEFAULT_POLICIES = ["agent", "rule", "sma20"]


@dataclass
class BaselineConfig:
    tickers: List[str]
    period: str
    policies: List[str]
    offline: bool = True
    use_ml_model: bool = False
    outdir: str = str(Path("cache") / "metrics" / "accuracy_baseline")
    quiet: bool = True
    fee_bps: float = 5.0
    slip_bps: float = 5.0
    walk_forward: bool = False
    wf_splits: int = 3
    tune_thresholds: bool = False


def run_baseline(cfg: BaselineConfig) -> Dict[str, dict]:
    t0 = time.time()
    out_path = Path(cfg.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, dict] = {}
    broker = BrokerConfig(fee_bps=cfg.fee_bps, slip_bps=cfg.slip_bps)

    rows: List[dict] = []
    for ticker in cfg.tickers:
        for policy in cfg.policies:
            if cfg.quiet:
                # Suppress verbose prints from agents during large runs
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    res = simulate(
                        ticker=ticker,
                        period=cfg.period,
                        broker=broker,
                        use_ml_model=cfg.use_ml_model,
                        offline=cfg.offline,
                        policy=policy,
                        outdir=None,
                        walk_forward=cfg.walk_forward,
                        wf_splits=cfg.wf_splits,
                        tune_thresholds=cfg.tune_thresholds,
                    )
            else:
                res = simulate(
                    ticker=ticker,
                    period=cfg.period,
                    broker=broker,
                    use_ml_model=cfg.use_ml_model,
                    offline=cfg.offline,
                    policy=policy,
                    outdir=None,
                    walk_forward=cfg.walk_forward,
                    wf_splits=cfg.wf_splits,
                    tune_thresholds=cfg.tune_thresholds,
                )
            m = res.get("metrics", {})
            row = {
                "ticker": ticker,
                "period": cfg.period,
                "policy": policy,
                **{k: float(m.get(k, 0.0)) for k in ("total_return", "cagr", "sharpe", "max_drawdown", "hit_rate")},
            }
            rows.append(row)
            results[f"{ticker}:{policy}"] = row

    df = pd.DataFrame(rows)
    csv_path = out_path / f"baseline_{cfg.period}.csv"
    df.to_csv(csv_path, index=False)

    # Extra metrics: turnover and cost drag if available from runs
    avg_turnover = float(df.get("turnover", pd.Series([0.0]*len(df))).mean()) if not df.empty else 0.0
    avg_cost_drag = float(df.get("cost_drag", pd.Series([0.0]*len(df))).mean()) if not df.empty else 0.0

    summary = {
        "tickers": cfg.tickers,
        "period": cfg.period,
        "policies": cfg.policies,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 2),
        "rows": rows,
        "config": {
            "fee_bps": cfg.fee_bps,
            "slip_bps": cfg.slip_bps,
            "walk_forward": cfg.walk_forward,
            "wf_splits": cfg.wf_splits,
            "tune_thresholds": cfg.tune_thresholds,
        },
        "extras": {
            "avg_turnover": avg_turnover,
            "avg_cost_drag": avg_cost_drag,
        },
        "by_policy": {
            p: {
                "avg_cagr": float(df[df["policy"] == p]["cagr"].mean() if not df.empty else 0.0),
                "avg_sharpe": float(df[df["policy"] == p]["sharpe"].mean() if not df.empty else 0.0),
                "avg_hit_rate": float(df[df["policy"] == p]["hit_rate"].mean() if not df.empty else 0.0),
            } for p in cfg.policies
        },
    }
    json_path = out_path / f"baseline_{cfg.period}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"csv": str(csv_path), "json": str(json_path), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run accuracy baseline backtests")
    parser.add_argument("--tickers", help="Comma-separated list of tickers", default=",")
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--online", action="store_true", help="Allow LLM calls")
    parser.add_argument("--ml", action="store_true", help="Use ML model when available")
    parser.add_argument("--outdir", default=str(Path("cache") / "metrics" / "accuracy_baseline"))
    parser.add_argument("--verbose", action="store_true", help="Show verbose agent logs during runs")
    # New flags
    parser.add_argument("--fee-bps", type=float, default=5.0, help="Commission per trade in basis points")
    parser.add_argument("--slip-bps", type=float, default=5.0, help="Slippage per trade in basis points")
    parser.add_argument("--walk-forward", action="store_true", help="Enable walk-forward evaluation")
    parser.add_argument("--wf-splits", type=int, default=3, help="Number of walk-forward splits")
    parser.add_argument("--tune-thresholds", action="store_true", help="Tune decision thresholds on validation folds")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in (args.tickers or "").split(",") if t.strip()] or DEFAULT_TICKERS
    policies = [p.strip() for p in (args.policies or "").split(",") if p.strip()] or DEFAULT_POLICIES
    cfg = BaselineConfig(
        tickers=tickers,
        period=args.period,
        policies=policies,
        offline=(not args.online),
        use_ml_model=bool(args.ml),
        outdir=args.outdir,
        quiet=(not args.verbose),
        fee_bps=args.fee_bps,
        slip_bps=args.slip_bps,
        walk_forward=bool(args.walk_forward),
        wf_splits=int(args.wf_splits),
        tune_thresholds=bool(args.tune_thresholds),
    )
    out = run_baseline(cfg)
    print(f"Baseline written: {out['csv']} | {out['json']}")


if __name__ == "__main__":
    main()


