import os

import pytest

from langgraph_flow import run_prediction


@pytest.mark.parametrize("ticker", ["AAPL"])  # keep it light
def test_ml_mode_runs_offline(ticker):
    os.environ["DISABLE_LLM"] = "1"
    # Even if the ML model path fails, pipeline should complete gracefully
    result = run_prediction(ticker, timeframe="1d", low_api_mode=True, fast_ta_mode=True, use_ml_model=True)
    assert isinstance(result, dict)
    # Should not crash; allow either success or presence of prediction output
    assert result.get("status") in {"success", "initialized", None} or "prediction_result" in result


