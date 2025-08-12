import math

from portfolio.engine import PortfolioConfig, compute_target_weights, allowable_buy_notional


def test_compute_target_weights_basic_long_only():
    cfg = PortfolioConfig(max_weight_per_ticker=0.10, max_gross_exposure=0.20, sizing_gain=0.10, allow_shorts=False)
    inputs = {
        "AAPL": {"prediction_result": {"direction": "UP", "confidence": 80}, "confidence_metrics": {"overall_confidence": 75}},
        "MSFT": {"prediction_result": {"direction": "DOWN", "confidence": 90}, "confidence_metrics": {"overall_confidence": 85}},
    }
    w = compute_target_weights(inputs, cfg)
    # Long-only clamps negative to zero
    assert w["MSFT"] == 0.0
    # AAPL gets positive weight scaled by sizing_gain and capped by per-name and gross limits
    assert 0.0 < w["AAPL"] <= cfg.max_weight_per_ticker
    assert math.isclose(sum(abs(x) for x in w.values()), min(cfg.max_gross_exposure, cfg.max_weight_per_ticker), rel_tol=1e-6) or sum(abs(x) for x in w.values()) <= cfg.max_gross_exposure


def test_compute_target_weights_gross_normalization():
    cfg = PortfolioConfig(max_weight_per_ticker=0.10, max_gross_exposure=0.15, sizing_gain=0.10, allow_shorts=False)
    inputs = {
        "A": {"prediction_result": {"direction": "UP", "confidence": 100}, "confidence_metrics": {"overall_confidence": 95}},
        "B": {"prediction_result": {"direction": "UP", "confidence": 100}, "confidence_metrics": {"overall_confidence": 95}},
    }
    w = compute_target_weights(inputs, cfg)
    total = sum(abs(x) for x in w.values())
    assert total <= cfg.max_gross_exposure + 1e-9


def test_allowable_buy_notional():
    cfg = PortfolioConfig(max_cash_utilization_per_day=0.25)
    assert allowable_buy_notional(10_000.0, cfg) == 2500.0
    assert allowable_buy_notional(0.0, cfg) == 0.0


