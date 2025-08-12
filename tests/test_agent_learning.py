from __future__ import annotations

from utils.agent_learn import LearnConfig, learn_once


def test_learn_once_runs_offline():
    cfg = LearnConfig(ticker="AAPL", timeframe="1d", period="6mo", iterations=5, learning_rate=0.05, offline=True)
    res = learn_once(cfg)
    assert isinstance(res, dict)
    assert "status" in res


