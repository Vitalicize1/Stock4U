#!/usr/bin/env python3
"""
Test: sentiment→technical bias wiring

Validates that SentimentIntegratorAgent writes adjusted trading signals
back into the technical analysis block so downstream consumers see the
biased recommendation and numeric strength.
"""

from agents.sentiment_integrator import SentimentIntegratorAgent


def main():
    agent = SentimentIntegratorAgent()

    # Case 1: Positive sentiment upgrades BUY→STRONG_BUY and nudges strength up
    technical = {
        "trading_signals": {
            "signals": [{"type": "BUY", "indicator": "RSI", "strength": "moderate"}],
            "signal_strength": 1.0,
            "overall_recommendation": "BUY",
        },
        "technical_score": 55,
    }
    sentiment_pos = {
        "overall_sentiment": {"sentiment_score": 0.6, "sentiment_label": "very_positive", "confidence": 0.8}
    }
    res_pos = agent.integrate_sentiment({
        "technical_analysis": technical,
        "sentiment_analysis": sentiment_pos,
        "data": {},
    })
    assert res_pos.get("status") == "success", f"integration failed: {res_pos}"
    assert technical["trading_signals"]["overall_recommendation"] == "STRONG_BUY", technical
    assert technical["trading_signals"]["signal_strength"] > 1.0, technical

    # Case 2: Negative sentiment upgrades SELL→STRONG_SELL and nudges strength down
    technical2 = {
        "trading_signals": {
            "signals": [{"type": "SELL", "indicator": "RSI", "strength": "moderate"}],
            "signal_strength": -1.0,
            "overall_recommendation": "SELL",
        },
        "technical_score": 45,
    }
    sentiment_neg = {
        "overall_sentiment": {"sentiment_score": -0.7, "sentiment_label": "very_negative", "confidence": 0.9}
    }
    res_neg = agent.integrate_sentiment({
        "technical_analysis": technical2,
        "sentiment_analysis": sentiment_neg,
        "data": {},
    })
    assert res_neg.get("status") == "success", f"integration failed: {res_neg}"
    assert technical2["trading_signals"]["overall_recommendation"] == "STRONG_SELL", technical2
    assert technical2["trading_signals"]["signal_strength"] < -1.0, technical2

    print("test_sentiment_bias: success")


if __name__ == "__main__":
    main()


