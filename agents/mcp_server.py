"""MCP server exposing Stock4U tools via Model Context Protocol.

This module provides a stdio MCP server exposing a small set of tools that
wrap the existing Stock4U workflow. It does not affect the dashboard/CLI.
"""

from typing import Dict, Any, List
from datetime import datetime
import os
import sys
from contextlib import redirect_stdout
import concurrent.futures as _futures

import pandas as pd
import yfinance as yf

# Ensure project root is on sys.path when launched via file path (e.g., mcp dev)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langgraph_flow import run_prediction
from utils.result_cache import get_cached_result as _get_cache, set_cached_result as _set_cache, invalidate_cached_result as _invalidate_cache

try:
    # FastMCP is the ergonomic server from the official MCP Python SDK
    from mcp.server.fastmcp import FastMCP, Context
except Exception as import_error:  # pragma: no cover - only if MCP is not installed
    FastMCP = None  # type: ignore
    Context = object  # type: ignore


def _serialize_dataframe(df: pd.DataFrame, max_rows: int = 200) -> Dict[str, Any]:
    """Convert a DataFrame to a JSON-serializable dict with safe truncation."""
    if df is None or getattr(df, "empty", True):
        return {"rows": [], "columns": []}
    trimmed = df.tail(max_rows).reset_index()
    try:
        # Convert Timestamp/Index to ISO strings when possible
        for col in trimmed.columns:
            if str(trimmed[col].dtype).startswith("datetime"):
                trimmed[col] = trimmed[col].dt.tz_localize(None).dt.isoformat()
    except Exception:
        pass
    return {
        "rows": trimmed.to_dict(orient="records"),
        "columns": [str(c) for c in trimmed.columns],
    }


# --------------------
# Tools (pure helpers)
# --------------------
def _get_timeout_env(var_name: str, default_seconds: int) -> int:
    try:
        val = int(os.getenv(var_name, str(default_seconds)))
        return max(1, val)
    except Exception:
        return default_seconds


def _run_with_timeout(fn, timeout_seconds: int) -> Dict[str, Any]:
    """Execute fn() with a timeout; return structured error on timeout."""
    try:
        with _futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(fn)
            return fut.result(timeout=timeout_seconds)
    except _futures.TimeoutError:
        return {"status": "error", "error": f"timeout after {timeout_seconds}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def fetch_stock_history_json(ticker: str, period: str = "1mo") -> Dict[str, Any]:
    """Fetch OHLCV history using yfinance and return a JSON-safe payload."""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return {
            "status": "success",
            "ticker": ticker,
            "period": period,
            "data": _serialize_dataframe(data),
        }
    except Exception as e:
        return {
            "status": "error",
            "ticker": ticker,
            "period": period,
            "error": str(e),
        }


def run_prediction_wrapper(
    ticker: str,
    timeframe: str = "1d",
    low_api_mode: bool = False,
    fast_ta_mode: bool = False,
    use_ml_model: bool = False,
) -> Dict[str, Any]:
    """Call the existing LangGraph workflow and return its result payload."""
    try:
        # Redirect stdout to stderr to avoid corrupting MCP stdio stream
        with redirect_stdout(sys.stderr):
            result = run_prediction(
                ticker=ticker,
                timeframe=timeframe,
                low_api_mode=low_api_mode,
                fast_ta_mode=fast_ta_mode,
                use_ml_model=use_ml_model,
            )
        # Ensure minimal stable keys
        return {
            "status": result.get("status", "success"),
            "ticker": ticker,
            "timeframe": timeframe,
            "result": result,
        }
    except Exception as e:
        return {
            "status": "error",
            "ticker": ticker,
            "timeframe": timeframe,
            "error": str(e),
        }


# --------------------
# MCP Server (FastMCP)
# --------------------
def _register_tools(app: Any) -> None:
    @app.tool()
    def get_stock_data(
        ticker: str,
        period: str = "1mo",
    ) -> Dict[str, Any]:
        """Return OHLCV history for a ticker. period examples: 5d, 1mo, 3mo, 1y."""
        timeout_s = _get_timeout_env("STOCK4U_MCP_TIMEOUT_STOCKDATA", 20)
        return _run_with_timeout(lambda: fetch_stock_history_json(ticker=ticker, period=period), timeout_s)

    @app.tool()
    def run_stock_prediction(
        ticker: str,
        timeframe: str = "1d",
        low_api_mode: bool = False,
        fast_ta_mode: bool = False,
        use_ml_model: bool = False,
    ) -> Dict[str, Any]:
        """Run the full prediction workflow and return structured results."""
        timeout_s = _get_timeout_env("STOCK4U_MCP_TIMEOUT_PREDICTION", 90)
        return _run_with_timeout(
            lambda: run_prediction_wrapper(
                ticker=ticker,
                timeframe=timeframe,
                low_api_mode=low_api_mode,
                fast_ta_mode=fast_ta_mode,
                use_ml_model=use_ml_model,
            ),
            timeout_s,
        )

    @app.tool()
    def ping() -> Dict[str, Any]:
        """Simple health check."""
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    @app.tool()
    def get_market_snapshot(ticker: str) -> Dict[str, Any]:
        """Lightweight summary: last close, change %, volume, and basic indicators (RSI, SMA20/50)."""
        def _compute() -> Dict[str, Any]:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")
                if hist is None or hist.empty:
                    return {"status": "error", "error": f"No data for {ticker}"}
                last = hist.tail(2)
                last_close = float(last["Close"].iloc[-1])
                prev_close = float(last["Close"].iloc[-2]) if len(last) > 1 else last_close
                change_pct = (last_close - prev_close) / prev_close * 100.0 if prev_close else 0.0
                volume = int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else None

                # Basic indicators from recent window
                window = hist.tail(50).copy()
                rsi = None
                sma20 = None
                sma50 = None
                try:
                    delta = window["Close"].diff()
                    gain = delta.clip(lower=0).rolling(14).mean()
                    loss = (-delta.clip(upper=0)).rolling(14).mean()
                    rs = (gain / (loss.replace(0, float("nan"))))
                    rsi = float(100 - (100 / (1 + rs.iloc[-1]))) if rs.notna().iloc[-1] else None
                except Exception:
                    pass
                try:
                    sma20 = float(window["Close"].rolling(20).mean().iloc[-1])
                except Exception:
                    pass
                try:
                    sma50 = float(window["Close"].rolling(50).mean().iloc[-1])
                except Exception:
                    pass

                return {
                    "status": "success",
                    "ticker": ticker,
                    "last_close": last_close,
                    "change_pct": change_pct,
                    "volume": volume,
                    "indicators": {"rsi": rsi, "sma_20": sma20, "sma_50": sma50},
                }
            except Exception as e:
                return {"status": "error", "ticker": ticker, "error": str(e)}

        timeout_s = _get_timeout_env("STOCK4U_MCP_TIMEOUT_SNAPSHOT", 15)
        return _run_with_timeout(_compute, timeout_s)

    @app.tool()
    def get_cached_result(cache_key: str, ttl_seconds: int = 900) -> Dict[str, Any]:
        """Return a cached result by key if fresh within ttl_seconds; else null data."""
        def _do():
            try:
                data = _get_cache(cache_key, ttl_seconds=ttl_seconds)
                return {"status": "success", "cache_key": cache_key, "data": data}
            except Exception as e:
                return {"status": "error", "cache_key": cache_key, "error": str(e)}
        timeout_s = _get_timeout_env("STOCK4U_MCP_TIMEOUT_CACHE", 5)
        return _run_with_timeout(_do, timeout_s)

    @app.tool()
    def invalidate_cache(cache_key: str) -> Dict[str, Any]:
        """Invalidate a cached result by key. Returns whether a file was removed."""
        def _do():
            try:
                removed = _invalidate_cache(cache_key)
                return {"status": "success", "cache_key": cache_key, "removed": bool(removed)}
            except Exception as e:
                return {"status": "error", "cache_key": cache_key, "error": str(e)}
        timeout_s = _get_timeout_env("STOCK4U_MCP_TIMEOUT_CACHE", 5)
        return _run_with_timeout(_do, timeout_s)


# Create a global server instance for MCP dev harness discovery
if FastMCP is not None:
    mcp = FastMCP(
        name="Stock4U MCP Server",
        instructions=(
            "Expose Stock4U analysis tools via Model Context Protocol (stdio server)."
        ),
    )
    _register_tools(mcp)
else:
    mcp = None  # type: ignore


def build_mcp_server() -> Any:
    """Return the global MCP app, or raise if MCP is not installed."""
    if mcp is None:
        raise RuntimeError(
            "The 'mcp' package is not installed. Please install with: pip install \"mcp[cli]\""
        )
    return mcp


def main() -> None:
    server = build_mcp_server()
    # Do NOT print to stdout; it will corrupt the MCP protocol. Use stderr.
    print("Starting Stock4U MCP server (stdio)...", file=sys.stderr, flush=True)
    # By default, FastMCP.run() starts a stdio server suitable for MCP clients
    try:
        server.run()
    except Exception as e:
        print(f"MCP server exited with error: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
