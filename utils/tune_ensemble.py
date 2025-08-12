from __future__ import annotations

"""
Hyperparameter tuning for the ensemble predictor.

Optimizes ensemble weights (llm, ml, rule) and optional Platt parameters (a, b)
per ticker/timeframe by minimizing log-loss on a bootstrap of recent samples.

Writes results to cache keys the ensemble already reads:
  - ensemble_weights::{ticker}::{timeframe}
  - ensemble_platt::{ticker}::{timeframe}
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import random
import time

import numpy as np

from utils.result_cache import set_cached_result, get_cached_result
from agents.tools.prediction_agent_tools import _direction_to_proba_up
from backtesting.run import simulate, BrokerConfig


@dataclass
class TuneConfig:
    ticker: str
    timeframe: str = "1d"
    period: str = "1y"
    trials: int = 50
    use_ml_model: bool = False
    offline: bool = True
    cv_splits: int = 3
    seed: int = 42


def _collect_samples(cfg: TuneConfig) -> List[Tuple[float, float, float, int]]:
    """Collect synthetic component probabilities and outcomes via simulate().

    This uses the simulate loop to harvest per-day direction decisions. We
    approximate component probabilities by calling the ensemble's components
    individually via existing tools is heavy; instead, we extract the agent's
    `prediction_result`, map to proba via `_direction_to_proba_up`, and derive
    simple rule/sma signals as additional channels to tune weights against.
    """
    # Use the backtesting simulate to roll daily; but here we only need
    # direction vs. simple baselines. We'll reuse simulate with policy paths.
    # Get agent decisions
    broker = BrokerConfig()
    res_agent = simulate(cfg.ticker, cfg.period, broker, use_ml_model=cfg.use_ml_model, offline=cfg.offline, policy="agent", outdir=None)
    res_rule = simulate(cfg.ticker, cfg.period, broker, use_ml_model=False, offline=True, policy="rule", outdir=None)
    res_sma = simulate(cfg.ticker, cfg.period, broker, use_ml_model=False, offline=True, policy="sma20", outdir=None)

    # Convert trades back into a rough daily direction; fallback to 0.5 if unknown
    def equity_to_labels(equity: List[float]) -> List[int]:
        labels: List[int] = []
        for i in range(1, len(equity)):
            labels.append(1 if equity[i] >= equity[i-1] else 0)
        return labels

    # Agent prediction result contains only final metrics; for simplicity, we will
    # derive outcomes from next-day equity change. This is a rough proxy, but good
    # enough for tuning default weights.
    y = equity_to_labels(res_agent["equity_curve"])  # 1 if up, 0 if down

    # Synthesize probabilities for channels using cumulative performance proxies
    # Normalize to [0,1]
    def perf_to_prob(curve: List[float]) -> List[float]:
        arr = np.asarray(curve, dtype=float)
        if len(arr) < 2:
            return [0.5]
        rets = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
        # map daily return to proba via sigmoid-like transform
        pr = 0.5 + np.tanh(rets * 10) / 2.0
        return pr.tolist()

    p_agent = perf_to_prob(res_agent["equity_curve"])  # len-1
    p_rule = perf_to_prob(res_rule["equity_curve"])[:len(p_agent)]
    p_sma = perf_to_prob(res_sma["equity_curve"])[:len(p_agent)]
    y = y[:len(p_agent)]

    samples: List[Tuple[float, float, float, int]] = []
    for i in range(len(y)):
        samples.append((p_agent[i], p_rule[i], p_sma[i], y[i]))
    return samples


def _logloss(p: float, y: int) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return - (y * np.log(p) + (1 - y) * np.log(1 - p))


def _evaluate(weights: Tuple[float, float, float], a: float, b: float, samples: List[Tuple[float, float, float, int]]) -> float:
    w_agent, w_rule, w_sma = weights
    wsum = max(1e-9, w_agent + w_rule + w_sma)
    w_agent, w_rule, w_sma = w_agent/wsum, w_rule/wsum, w_sma/wsum
    loss = 0.0
    for pa, pr, ps, y in samples:
        p = w_agent * pa + w_rule * pr + w_sma * ps
        # optional calibration
        p_cal = 1.0 / (1.0 + np.exp(-(a * p + b)))
        loss += _logloss(p_cal, y)
    return loss / float(len(samples) or 1)


def _ts_cv_loss(weights: Tuple[float, float, float], a: float, b: float, samples: List[Tuple[float, float, float, int]], splits: int) -> float:
    if splits <= 1 or len(samples) < (splits + 1) * 10:
        return _evaluate(weights, a, b, samples)
    n = len(samples)
    fold = n // (splits + 1)
    losses: List[float] = []
    for k in range(1, splits + 1):
        train_end = fold * k
        val_end = fold * (k + 1)
        val = samples[train_end:val_end]
        if not val:
            continue
        losses.append(_evaluate(weights, a, b, val))
    return float(sum(losses) / max(1, len(losses)))


def tune(cfg: TuneConfig) -> Dict[str, Any]:
    samples = _collect_samples(cfg)
    if len(samples) < 10:
        return {"status": "error", "error": "insufficient samples"}

    rng = random.Random(cfg.seed)
    best = {"loss": float("inf"), "weights": (0.5, 0.3, 0.2), "a": 1.0, "b": 0.0}
    for _ in range(cfg.trials):
        # Random search in a reasonable range
        w = (rng.random(), rng.random(), rng.random())
        a = rng.uniform(0.5, 2.0)
        b = rng.uniform(-1.0, 1.0)
        loss = _ts_cv_loss(w, a, b, samples, cfg.cv_splits)
        if loss < best["loss"]:
            best = {"loss": loss, "weights": w, "a": a, "b": b}

    w_agent, w_rule, w_sma = best["weights"]
    wsum = max(1e-9, w_agent + w_rule + w_sma)
    weights_norm = {"llm": 0.5, "ml": 0.4, "rule": 0.1}
    # Map: agent≈llm, rule≈rule, sma≈ml surrogate here (just to fill 3 channels consistently)
    weights_norm = {
        "llm": float(w_agent/wsum),
        "rule": float(w_rule/wsum),
        "ml": float(w_sma/wsum),
        "auc_raw": 0.0,  # placeholder; not computed in this quick tuner
        "n_samples": len(samples),
    }

    # Persist to cache keys used by ensemble (with regression guard)
    key_w = f"ensemble_weights::{cfg.ticker}::{cfg.timeframe}"
    key_p = f"ensemble_platt::{cfg.ticker}::{cfg.timeframe}"
    prev = get_cached_result(key_w, ttl_seconds=365 * 24 * 3600) or {}
    prev_loss = None
    try:
        # if previous manifest exists, load its loss
        manifest_dir = Path("cache") / "metrics" / "ensemble_tuning"
        mpath = manifest_dir / f"best_{cfg.ticker}_{cfg.timeframe}.json"
        if mpath.exists():
            import json as _json
            with open(mpath, "r", encoding="utf-8") as f:
                prev_manifest = _json.load(f)
                prev_loss = float(prev_manifest.get("loss", 0.0))
    except Exception:
        prev_loss = None

    # If previous loss exists and new one is worse by >5%, skip update
    if prev_loss is not None and best["loss"] > prev_loss * 1.05:
        return {"status": "skipped", "reason": "regression_guard", "prev_loss": prev_loss, "new_loss": best["loss"]}

    set_cached_result(key_w, weights_norm)
    set_cached_result(key_p, {"a": float(best["a"]), "b": float(best["b"])})

    # Write manifest artifact
    try:
        manifest_dir = Path("cache") / "metrics" / "ensemble_tuning"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "ticker": cfg.ticker,
            "timeframe": cfg.timeframe,
            "period": cfg.period,
            "trials": cfg.trials,
            "cv_splits": cfg.cv_splits,
            "seed": cfg.seed,
            "loss": float(best["loss"]),
            "weights": weights_norm,
            "platt": {"a": float(best["a"]), "b": float(best["b"])},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        import json as _json
        with open(manifest_dir / f"best_{cfg.ticker}_{cfg.timeframe}.json", "w", encoding="utf-8") as f:
            _json.dump(manifest, f, indent=2)
    except Exception:
        pass

    return {"status": "success", "weights": weights_norm, "platt": {"a": best["a"], "b": best["b"]}, "loss": best["loss"]}


def main() -> None:
    p = argparse.ArgumentParser(description="Tune ensemble weights and calibration")
    p.add_argument("--ticker", required=True)
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--period", default="1y")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--cv-splits", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ml", action="store_true")
    p.add_argument("--online", action="store_true")
    args = p.parse_args()

    cfg = TuneConfig(
        ticker=args.ticker.upper(),
        timeframe=args.timeframe,
        period=args.period,
        trials=int(args.trials),
        cv_splits=int(args["cv_splits"]) if isinstance(args, dict) else int(args.cv_splits),
        seed=int(args["seed"]) if isinstance(args, dict) else int(args.seed),
        use_ml_model=bool(args.ml),
        offline=(not args.online),
    )

    res = tune(cfg)
    if res.get("status") == "success":
        print("Best weights:", res["weights"])
        print("Platt params:", res["platt"])
        print("Loss:", round(float(res["loss"]), 6))
    else:
        print("Tuning failed:", res.get("error"))


if __name__ == "__main__":
    main()


