from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LinearModel:
    weights: np.ndarray
    bias: float

    def predict_proba_up(self, X: np.ndarray) -> float:
        z = float(np.dot(self.weights, X) + self.bias)
        # Sigmoid
        return 1.0 / (1.0 + np.exp(-z))


def load_default_model(n_features: int) -> LinearModel:
    # Simple heuristic weights: emphasize RSI, trend strength, integrated score
    w = np.zeros(n_features, dtype=float)
    # Attempt to align with features order in features.py
    # [rsi, macd, sma_20, sma_50, sma_200, trend_strength, adx_strength, signal_strength, integrated_score, daily_return_pct]
    idx = {
        0: 0.02,   # rsi
        1: 0.01,   # macd
        2: 0.0,    # sma_20
        3: 0.0,    # sma_50
        4: 0.0,    # sma_200
        5: 0.03,   # trend_strength
        6: 0.02,   # adx_strength
        7: 0.03,   # signal_strength
        8: 0.03,   # integrated_score
        9: 0.02,   # daily_return_pct
    }
    for i, v in idx.items():
        if i < n_features:
            w[i] = v
    b = -2.0
    return LinearModel(weights=w, bias=b)


