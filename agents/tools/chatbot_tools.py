"""
Chatbot Tools

Small toolized utilities that help parse intent, extract tickers, and format
responses for the chatbot agent. These are deterministic and LLM-free.
"""

from typing import Dict, Any, Optional
import re
from langchain_core.tools import tool


@tool
def parse_user_intent(user_input: str) -> Dict[str, Any]:
    """
    Parse user input to determine intent (stock_analysis, web_search, workflow_info, help, greeting)
    and extract a candidate ticker if present.
    """
    text = (user_input or "").strip()
    low = text.lower()
    def _is_stock():
        return any(k in low for k in ["analyze", "stock", "prediction", "price", "forecast", "analysis"]) 
    def _is_search():
        return any(k in low for k in ["search", "find", "look up", "latest news", "recent"]) 
    def _is_workflow():
        return any(k in low for k in ["workflow", "agents", "process", "how does it work", "pipeline"]) 
    def _is_help():
        return any(k in low for k in ["help", "what can you do", "capabilities", "examples", "guide"]) 

    # Extract ticker: first 1-5 letter uppercase word
    ticker: Optional[str] = None
    for word in text.split():
        clean = re.sub(r"[^A-Z]", "", word)
        if 1 <= len(clean) <= 5 and clean.isupper():
            # Validate the extracted ticker
            from utils.validation import InputValidator
            validation_result = InputValidator.validate_ticker_symbol(clean)
            if validation_result.is_valid:
                ticker = validation_result.sanitized_value
                break

    intent = "greeting"
    if _is_stock():
        intent = "stock_analysis"
    elif _is_search():
        intent = "web_search"
    elif _is_workflow():
        intent = "workflow_info"
    elif _is_help():
        intent = "help"

    return {"status": "success", "intent": intent, "ticker": ticker}


@tool
def format_stock_summary_for_chat(ticker: str, final_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a concise markdown string from a final_summary payload for chat display.
    """
    try:
        ps = (final_summary or {}).get("prediction_summary", {})
        ts = (final_summary or {}).get("technical_summary", {})
        es = (final_summary or {}).get("evaluation_summary", {})
        rec = (final_summary or {}).get("final_recommendation", {})

        direction = str(ps.get("direction", "NEUTRAL")).upper()
        conf = float(ps.get("confidence", 50))
        overall = (ts.get("trading_signals", {}) or {}).get("overall_recommendation", "HOLD")
        score = float(es.get("overall_score", 0))
        action = rec.get("action", "HOLD")

        text = (
            f"📊 {ticker} — Direction: {direction}, Confidence: {conf:.1f}%\n"
            f"🔧 Technical: {overall}, Eval Score: {score:.1f}/100\n"
            f"💡 Recommendation: {action}"
        )
        return {"status": "success", "text": text}
    except Exception as e:
        return {"status": "error", "error": f"format_stock_summary_for_chat failed: {e}", "text": ""}


