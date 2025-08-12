from __future__ import annotations

"""
Nightly Agent Learning batch runner

Runs `utils.agent_learn.learn_once` across a set of tickers and timeframes
and writes a consolidated summary artifact under
  cache/metrics/agent_learning/learning_YYYYmmdd_HHMMSS.json

This module is designed to be executed by an OS scheduler (e.g., Windows Task
Scheduler or cron). It performs a single pass and exits.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from utils.agent_learn import LearnConfig, learn_once


DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
DEFAULT_TIMEFRAMES = ["1d", "1w"]


@dataclass
class BatchConfig:
    tickers: List[str]
    timeframes: List[str]
    period: str = "1y"
    iterations: int = 100
    lr: float = 0.1
    use_ml_model: bool = False
    offline: bool = True
    outdir: str = str(Path("cache") / "metrics" / "agent_learning")


def run_learning_batch(cfg: BatchConfig) -> Dict[str, dict]:
    t0 = time.time()
    out_path = Path(cfg.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: Dict[str, dict] = {}
    for ticker in cfg.tickers:
        for tf in cfg.timeframes:
            res = learn_once(
                LearnConfig(
                    ticker=ticker.upper(),
                    timeframe=tf,
                    period=cfg.period,
                    iterations=int(cfg.iterations),
                    learning_rate=float(cfg.lr),
                    use_ml_model=bool(cfg.use_ml_model),
                    offline=bool(cfg.offline),
                )
            )
            results[f"{ticker}:{tf}"] = res

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 2),
        "tickers": cfg.tickers,
        "timeframes": cfg.timeframes,
        "period": cfg.period,
        "iterations": cfg.iterations,
        "lr": cfg.lr,
        "use_ml_model": cfg.use_ml_model,
        "offline": cfg.offline,
        "results": results,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_path / f"learning_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Update last_status pointer for easy health checks
    try:
        with open(out_path / "last_status.json", "w", encoding="utf-8") as f:
            json.dump({"artifact": str(json_path), "generated_at": summary["generated_at"], "elapsed_s": summary["elapsed_s"]}, f, indent=2)
    except Exception:
        pass

    return {"json": str(json_path), "summary": summary}


def main() -> None:
    p = argparse.ArgumentParser(description="Run nightly agent learning across tickers/timeframes")
    p.add_argument("--tickers", default=",")
    p.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    p.add_argument("--period", default="1y")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--ml", action="store_true")
    p.add_argument("--online", action="store_true")
    p.add_argument("--outdir", default=str(Path("cache") / "metrics" / "agent_learning"))
    args = p.parse_args()

    tickers = [t.strip().upper() for t in (args.tickers or "").split(",") if t.strip()] or DEFAULT_TICKERS
    timeframes = [tf.strip() for tf in (args.timeframes or "").split(",") if tf.strip()] or DEFAULT_TIMEFRAMES

    out = run_learning_batch(
        BatchConfig(
            tickers=tickers,
            timeframes=timeframes,
            period=args.period,
            iterations=int(args.iterations),
            lr=float(args.lr),
            use_ml_model=bool(args.ml),
            offline=(not args.online),
            outdir=args.outdir,
        )
    )
    print(f"Learning summary written: {out['json']}")


if __name__ == "__main__":
    main()


