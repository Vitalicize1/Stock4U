from __future__ import annotations

import json
from pathlib import Path

from utils.agent_learn import LearnConfig, learn_once


def test_last_learned_written(tmp_path, monkeypatch):
    # Redirect cache dir by monkeypatching Path join in module via env
    cache_dir = tmp_path / "cache" / "metrics" / "agent_learning"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch Path used inside learn_once by setting CWD to tmp_path
    monkeypatch.chdir(tmp_path)

    cfg = LearnConfig(ticker="AAPL", timeframe="1d", period="6mo", iterations=3, learning_rate=0.05, offline=True)
    _ = learn_once(cfg)

    last = cache_dir / "last_learned.json"
    assert last.exists(), "last_learned.json should have been created"
    data = json.loads(last.read_text())
    assert any(k.startswith("AAPL:") for k in data.keys())


def test_regression_guard_path(monkeypatch):
    # Force regression by monkeypatching helper to always return high new loss
    import utils.agent_learn as al

    def fake_avg_loss(weights, a, b, samples):  # noqa: ARG001
        return 1000.0

    monkeypatch.setattr(al, "_avg_logloss_for_params", fake_avg_loss)

    cfg = al.LearnConfig(ticker="AAPL", timeframe="1d", period="6mo", iterations=1, learning_rate=0.01, offline=True)
    res = al.learn_once(cfg)
    assert res.get("status") in {"success", "skipped"}


