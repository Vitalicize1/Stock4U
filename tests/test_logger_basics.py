from pathlib import Path
import json

from utils import logger as lg


def test_log_metric_and_alert(tmp_path, monkeypatch):
    metrics_file = tmp_path / "metrics.jsonl"
    alerts_file = tmp_path / "alerts.log"
    monkeypatch.setenv("METRICS_FILE", str(metrics_file))
    monkeypatch.setenv("ALERTS_FILE", str(alerts_file))
    monkeypatch.setenv("ALERT_ENABLED", "true")

    lg.log_metric("test_latency_ms", 123.0, {"k": "v"})
    with lg.time_block("block_ms"):
        pass
    lg.emit_alert("Test alert", "Something happened", "warning", {"a": "b"})

    # Metrics JSONL has lines
    lines = metrics_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    rec = json.loads(lines[0])
    assert rec["event"] == "metric" and rec["name"] == "test_latency_ms"

    # Alerts JSONL readable
    al = alerts_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(al) >= 1
    arec = json.loads(al[-1])
    assert arec["title"] == "Test alert"


