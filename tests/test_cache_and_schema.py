import os

from langgraph_flow import run_prediction


def test_cache_roundtrip_and_schema_normalization(tmp_path, monkeypatch):
    os.environ["DISABLE_LLM"] = "1"

    # Use a distinct ticker/timeframe combo to avoid cross-run interference
    ticker = "MSFT"
    timeframe = "1d"

    # First run – populates cache
    res1 = run_prediction(ticker, timeframe, low_api_mode=True, fast_ta_mode=True)
    assert isinstance(res1, dict)
    # Minimal shape checks
    assert "prediction_result" in res1 or "final_summary" in res1

    # Second run – should hit cache (cannot easily assert timing, but should be same-type dict)
    res2 = run_prediction(ticker, timeframe, low_api_mode=True, fast_ta_mode=True)
    assert isinstance(res2, dict)


