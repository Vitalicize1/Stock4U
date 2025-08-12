from __future__ import annotations

"""
Nightly QA script

Calls backend endpoints to validate agent outputs and ensure baseline
artifacts exist. Exits non-zero on schema errors or missing artifacts so it
can be used in CI or a scheduled task.
"""

import os
import sys
import time
import json
from typing import Any, Dict

import requests


API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TOKEN = os.getenv("API_TOKEN", "")
PERIOD = os.getenv("QA_BASELINE_PERIOD", "1y")
TICKERS = os.getenv("QA_TICKERS", "AAPL,MSFT,GOOGL")


def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def run_validation() -> Dict[str, Any]:
    r = requests.get(f"{API_URL}/validation/run", params={"tickers": TICKERS}, headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()


def get_baseline_latest() -> Dict[str, Any]:
    r = requests.get(f"{API_URL}/baseline/latest", params={"period": PERIOD}, headers=_headers(), timeout=30)
    if r.status_code == 404:
        return {"status": "missing"}
    r.raise_for_status()
    return r.json()


def main() -> None:
    start = time.time()
    failures: list[str] = []

    # 1) Validation run
    try:
        val = run_validation()
        report = val.get("report", {})
        for row in report.get("results", []):
            if row.get("status") != "success":
                failures.append(f"validation error: {row}")
            else:
                if row.get("errors"):
                    failures.append(f"schema errors for {row.get('ticker')}: {row.get('errors')}")
                # Fail on badge == fail
                if row.get("badge") == "fail":
                    failures.append(f"validation fail badge for {row.get('ticker')}")
    except Exception as e:
        failures.append(f"validation endpoint failed: {e}")

    # 2) Baseline artifact check
    try:
        base = get_baseline_latest()
        if base.get("status") == "missing":
            failures.append(f"baseline summary for period {PERIOD} not found")
    except Exception as e:
        failures.append(f"baseline latest failed: {e}")

    elapsed = time.time() - start
    summary = {"elapsed_s": round(elapsed, 2), "failures": failures}
    print(json.dumps(summary, indent=2))

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()


