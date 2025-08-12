from __future__ import annotations

"""
Agent Learning utilities

Per-ticker, per-timeframe lightweight online learning that updates the
ensemble weights (llm/ml/rule proxies) and optional Platt calibration
parameters. The learned parameters are persisted into the cache keys the
prediction agent already reads:
  - ensemble_weights::{ticker}::{timeframe}
  - ensemble_platt::{ticker}::{timeframe}

This module is intentionally simple and fast; it uses samples harvested via
the backtesting simulate loop and performs a few gradient steps on log-loss.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
from pathlib import Path
import json

import numpy as np

from backtesting.run import simulate, BrokerConfig
from utils.result_cache import get_cached_result, set_cached_result


def _regression(old_metrics: Dict[str, float], new_metrics: Dict[str, float]) -> bool:
    """Heuristic guardrail: flag regression if Sharpe drops >20% or hit rate drops >10pp."""
    try:
        old_sharpe = float(old_metrics.get("sharpe", 0.0))
        new_sharpe = float(new_metrics.get("sharpe", 0.0))
        old_hit = float(old_metrics.get("hit_rate", 0.0))
        new_hit = float(new_metrics.get("hit_rate", 0.0))
        sharpe_drop = (old_sharpe - new_sharpe) / (abs(old_sharpe) + 1e-9)
        hit_drop = old_hit - new_hit
        return (sharpe_drop > 0.2) or (hit_drop > 0.10)
    except Exception:
        return False


@dataclass
class LearnConfig:
    ticker: str
    timeframe: str = "1d"
    period: str = "1y"
    iterations: int = 100
    learning_rate: float = 0.1
    use_ml_model: bool = False
    offline: bool = True


def _collect_samples(ticker: str, period: str, use_ml_model: bool, offline: bool) -> List[Tuple[float, float, float, int]]:
    """Build training samples: (p_agent, p_rule, p_sma, y).

    Probabilities are derived from daily equity changes of each policy, mapped
    through a smooth transform. Target y is 1 if next-day equity increases.
    """
    broker = BrokerConfig()
    res_agent = simulate(ticker, period, broker, use_ml_model=use_ml_model, offline=offline, policy="agent", outdir=None)
    res_rule = simulate(ticker, period, broker, use_ml_model=False, offline=True, policy="rule", outdir=None)
    res_sma = simulate(ticker, period, broker, use_ml_model=False, offline=True, policy="sma20", outdir=None)

    def to_labels(curve: List[float]) -> List[int]:
        labels: List[int] = []
        for i in range(1, len(curve)):
            labels.append(1 if curve[i] >= curve[i - 1] else 0)
        return labels

    def perf_to_prob(curve: List[float]) -> List[float]:
        arr = np.asarray(curve, dtype=float)
        if len(arr) < 2:
            return [0.5]
        rets = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
        return (0.5 + np.tanh(rets * 10.0) / 2.0).tolist()

    y = to_labels(res_agent["equity_curve"])  # length N-1
    p_agent = perf_to_prob(res_agent["equity_curve"])  # length N-1
    p_rule = perf_to_prob(res_rule["equity_curve"])[: len(p_agent)]
    p_sma = perf_to_prob(res_sma["equity_curve"])[: len(p_agent)]
    y = y[: len(p_agent)]

    samples: List[Tuple[float, float, float, int]] = []
    for i in range(len(y)):
        samples.append((p_agent[i], p_rule[i], p_sma[i], y[i]))
    return samples


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _loss_and_grads(weights: np.ndarray, a: float, b: float, samples: List[Tuple[float, float, float, int]]):
    # Normalize weights to simplex
    w = np.maximum(1e-9, weights)
    w = w / w.sum()
    grad_w = np.zeros_like(w)
    grad_a = 0.0
    grad_b = 0.0
    loss = 0.0
    for pa, pr, ps, y in samples:
        p_vec = np.array([pa, pr, ps], dtype=float)
        p_mix = float(np.dot(w, p_vec))
        z = a * p_mix + b
        p = _sigmoid(z)
        # log-loss
        loss += - (y * np.log(max(p, 1e-9)) + (1 - y) * np.log(max(1 - p, 1e-9)))
        # gradients
        delta = p - y  # dL/dz
        grad_a += delta * p_mix
        grad_b += delta
        grad_w += delta * a * p_vec
    n = float(len(samples) or 1)
    return loss / n, grad_w / n, grad_a / n, grad_b / n, w


def _avg_logloss_for_params(weights: np.ndarray, a: float, b: float, samples: List[Tuple[float, float, float, int]]) -> float:
    w = np.maximum(1e-9, weights)
    w = w / w.sum()
    loss = 0.0
    for pa, pr, ps, y in samples:
        p_vec = np.array([pa, pr, ps], dtype=float)
        p_mix = float(np.dot(w, p_vec))
        z = a * p_mix + b
        p = _sigmoid(z)
        loss += - (y * np.log(max(p, 1e-9)) + (1 - y) * np.log(max(1 - p, 1e-9)))
    return float(loss / float(len(samples) or 1))


def learn_once(cfg: LearnConfig) -> Dict[str, float]:
    samples = _collect_samples(cfg.ticker, cfg.period, cfg.use_ml_model, cfg.offline)
    if len(samples) < 10:
        return {"status": "error", "error": "insufficient samples"}

    # Initialize from cache if present
    key_w = f"ensemble_weights::{cfg.ticker}::{cfg.timeframe}"
    key_p = f"ensemble_platt::{cfg.ticker}::{cfg.timeframe}"
    w_cached = get_cached_result(key_w, ttl_seconds=365 * 24 * 3600) or {}
    a_cached = get_cached_result(key_p, ttl_seconds=365 * 24 * 3600) or {}
    w0 = np.array([
        float(w_cached.get("llm", 0.5)),
        float(w_cached.get("rule", 0.1)),
        float(w_cached.get("ml", 0.4)),
    ], dtype=float)
    a = float(a_cached.get("a", 1.0))
    b = float(a_cached.get("b", 0.0))

    weights = w0.copy()
    lr = float(cfg.learning_rate)
    for _ in range(int(cfg.iterations)):
        loss, gw, ga, gb, w_norm = _loss_and_grads(weights, a, b, samples)
        weights = weights - lr * gw
        a = float(np.clip(a - lr * ga, 0.1, 3.0))
        b = float(np.clip(b - lr * gb, -2.0, 2.0))
        # small decay toward equal weights to avoid collapse
        weights = 0.98 * weights + 0.02 * np.array([1/3,1/3,1/3])
        # positivity
        weights = np.maximum(1e-6, weights)

    # Final normalized weights
    w_final = weights / weights.sum()
    out_weights = {
        "llm": float(w_final[0]),
        "rule": float(w_final[1]),
        "ml": float(w_final[2]),
        "n_samples": len(samples),
    }
    
    # Guardrail: revert if new parameters worsen average log-loss > 5%
    try:
        old_w = np.array([
            float(w_cached.get("llm", 0.5)),
            float(w_cached.get("rule", 0.1)),
            float(w_cached.get("ml", 0.4)),
        ], dtype=float)
        old_a = float(a_cached.get("a", 1.0))
        old_b = float(a_cached.get("b", 0.0))
        new_loss = _avg_logloss_for_params(w_final, a, b, samples)
        old_loss = _avg_logloss_for_params(old_w, old_a, old_b, samples)
        if new_loss > old_loss * 1.05:
            # Do not persist; return previous as effective
            return {"status": "skipped", "reason": "regression_guard", "prev_loss": old_loss, "new_loss": new_loss}
    except Exception:
        pass

    set_cached_result(key_w, out_weights)
    set_cached_result(key_p, {"a": float(a), "b": float(b)})

    # Write changelog entry
    try:
        log_dir = Path("cache") / "metrics" / "agent_learning"
        log_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "ticker": cfg.ticker,
            "timeframe": cfg.timeframe,
            "period": cfg.period,
            "iterations": cfg.iterations,
            "lr": cfg.learning_rate,
            "weights": out_weights,
            "platt": {"a": a, "b": b},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        ts = time.strftime("%Y%m%d_%H%M%S")
        with open(log_dir / f"changelog_{cfg.ticker}_{cfg.timeframe}_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
        # Update last_learned aggregator
        agg_path = log_dir / "last_learned.json"
        try:
            if agg_path.exists():
                with open(agg_path, "r", encoding="utf-8") as f:
                    agg = json.load(f)
            else:
                agg = {}
        except Exception:
            agg = {}
        agg[f"{cfg.ticker}:{cfg.timeframe}"] = entry
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
    except Exception:
        pass

    return {"status": "success", "weights": out_weights, "platt": {"a": a, "b": b}}


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Agent learning: update ensemble weights and calibration")
    p.add_argument("--ticker", required=True)
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--period", default="1y")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--ml", action="store_true")
    p.add_argument("--online", action="store_true")
    args = p.parse_args()

    cfg = LearnConfig(
        ticker=args.ticker.upper(),
        timeframe=args.timeframe,
        period=args.period,
        iterations=int(args.iterations),
        learning_rate=float(args.lr),
        use_ml_model=bool(args.ml),
        offline=(not args.online),
    )
    res = learn_once(cfg)
    print(res)


if __name__ == "__main__":
    main()


