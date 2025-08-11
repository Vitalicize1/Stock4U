import os

from langgraph_flow import run_prediction


def test_smoke_run_prediction_low_api_fast_ta():
    os.environ["DISABLE_LLM"] = "1"
    result = run_prediction("AAPL", timeframe="1d", low_api_mode=True, fast_ta_mode=True)

    # Basic contract checks
    assert isinstance(result, dict)
    assert result.get("status") in {"success", "initialized", None} or "prediction_result" in result
    # Minimal expected keys
    assert "technical_analysis" in result or "enhanced_technical_analysis" in result
    assert "prediction_result" in result or result.get("final_summary") is not None


