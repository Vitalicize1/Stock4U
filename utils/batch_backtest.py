from __future__ import annotations

"""
Batch backtesting CLI

Runs multiple backtests across a set of tickers and policies, aggregating
metrics to a single CSV and JSON summary under cache/results/batch_backtest/.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd

from backtesting.run import simulate, BrokerConfig


@dataclass
class BatchConfig:
    tickers: List[str]
    period: str = "1y"
    policies: List[str] = None  # e.g., ["agent","rule","sma20"]
    use_ml_model: bool = False
    offline: bool = True
    outdir: str = str(Path("cache") / "results" / "batch_backtest")
    max_workers: int = 4


def _run_one(ticker: str, policy: str, period: str, use_ml_model: bool, offline: bool) -> Dict[str, Any]:
    res = simulate(
        ticker=ticker,
        period=period,
        broker=BrokerConfig(),
        use_ml_model=use_ml_model,
        offline=offline,
        policy=policy,
        outdir=None,
    )
    m = res.get("metrics", {})
    return {
        "ticker": ticker,
        "period": period,
        "policy": policy,
        "total_return": float(m.get("total_return", 0.0)),
        "cagr": float(m.get("cagr", 0.0)),
        "sharpe": float(m.get("sharpe", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "hit_rate": float(m.get("hit_rate", 0.0)),
    }


def run_batch(cfg: BatchConfig) -> Dict[str, Any]:
    t0 = time.time()
    out_path = Path(cfg.outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    policies = cfg.policies or ["agent", "rule", "sma20"]

    futures = []
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        for t in cfg.tickers:
            for p in policies:
                futures.append(ex.submit(_run_one, t, p, cfg.period, cfg.use_ml_model, cfg.offline))
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({"error": str(e)})

    df = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_path / f"batch_{cfg.period}_{ts}.csv"
    df.to_csv(csv_path, index=False)
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 2),
        "tickers": cfg.tickers,
        "period": cfg.period,
        "policies": policies,
        "rows": rows,
        "by_policy": {
            p: {
                "avg_cagr": float(df[df["policy"] == p]["cagr"].mean() if not df.empty else 0.0),
                "avg_sharpe": float(df[df["policy"] == p]["sharpe"].mean() if not df.empty else 0.0),
                "avg_hit_rate": float(df[df["policy"] == p]["hit_rate"].mean() if not df.empty else 0.0),
            } for p in policies
        },
    }
    json_path = out_path / f"batch_{cfg.period}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"csv": str(csv_path), "json": str(json_path), "summary": summary}


def main() -> None:
    p = argparse.ArgumentParser(description="Run batch backtests across multiple tickers/policies")
    p.add_argument("--tickers", required=True, help="Comma-separated tickers")
    p.add_argument("--period", default="1y")
    p.add_argument("--policies", default="agent,rule,sma20")
    p.add_argument("--ml", action="store_true")
    p.add_argument("--online", action="store_true")
    p.add_argument("--outdir", default=str(Path("cache") / "results" / "batch_backtest"))
    p.add_argument("--max_workers", type=int, default=4)
    args = p.parse_args()

    cfg = BatchConfig(
        tickers=[t.strip().upper() for t in args.tickers.split(",") if t.strip()],
        period=args.period,
        policies=[p.strip() for p in (args.policies or "").split(",") if p.strip()],
        use_ml_model=bool(args.ml),
        offline=(not args.online),
        outdir=args.outdir,
        max_workers=int(args.max_workers),
    )

    out = run_batch(cfg)
    print(f"Batch artifacts: {out['csv']} | {out['json']}")


if __name__ == "__main__":
    main()


