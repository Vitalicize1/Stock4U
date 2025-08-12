from __future__ import annotations

"""Helpers to prepare demo-mode cached artifacts."""

from pathlib import Path
import json
from typing import Dict, Any


def save_demo_result(ticker: str, timeframe: str, normalized_result: Dict[str, Any]) -> str:
    out_dir = Path("cache") / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{ticker.upper()}_{timeframe}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(normalized_result, f, indent=2)
    return str(fp)


