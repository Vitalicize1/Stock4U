"""
Batch calibration runner.

Example:
  python -m utils.batch_calibrate --tickers AAPL,MSFT,NVDA --period 1y --offline --warmup 60 --show
"""

from __future__ import annotations

import argparse

from backtesting.calibrate import main as calibrate_main


def main() -> None:
    # Delegate to calibrate CLI; this module is a thin alias to make common usage memorable.
    calibrate_main()


if __name__ == "__main__":
    main()


