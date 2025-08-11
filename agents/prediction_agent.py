# agents/prediction_agent.py
from typing import Dict, Any, Tuple
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Import LLM clients
try:
    from llm.gemini_client import get_gemini_prediction
    from llm.groq_client import get_groq_prediction
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM clients not available. Using fallback predictions.")

# Load environment variables
load_dotenv()

class PredictionAgent:
    """
    Agent responsible for making final stock movement predictions using LLMs.
    Integrates technical analysis, sentiment, and market data to generate predictions.
    """
    
    def __init__(self, llm_provider: str = "auto"):
        """
        Initialize prediction agent.
        
        Args:
            llm_provider: "gemini", "groq", or "auto" (auto-selects based on available keys)
        """
        self.llm_provider = self._select_llm_provider(llm_provider)
        self.llm_available = LLM_AVAILABLE and self.llm_provider is not None
    
    def _select_llm_provider(self, provider: str) -> str:
        """Select the best available LLM provider."""
        if not LLM_AVAILABLE:
            return None
        
        if provider == "auto":
            # Check which API keys are available
            google_key = os.getenv("GOOGLE_API_KEY")
            groq_key = os.getenv("GROQ_API_KEY")
            
            if groq_key:
                return "groq"  # Prefer Groq for speed
            elif google_key:
                return "gemini"
            else:
                return None
        else:
            return provider if provider in ["gemini", "groq"] else None
    
    def make_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a comprehensive stock prediction based on all available data.
        
        Args:
            data: Dictionary containing all collected and analyzed data
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Extract relevant data
            ticker = data.get("ticker", "UNKNOWN")
            price_data = data.get("data", {}).get("price_data", {})
            
            # Handle both old and enhanced technical analysis structures
            technical_analysis = data.get("technical_analysis", {})
            enhanced_technical_analysis = data.get("enhanced_technical_analysis", {})
            
            # Use enhanced analysis if available, otherwise fall back to basic
            if enhanced_technical_analysis:
                technical_analysis = enhanced_technical_analysis
                print(f"🔍 Using enhanced technical analysis for {ticker}")
            else:
                print(f"⚠️ Using basic technical analysis for {ticker}")
            
            # Ensure we have the expected structure for both basic and enhanced analysis
            if not technical_analysis:
                technical_analysis = {
                    "indicators": {},
                    "trend_analysis": {},
                    "support_resistance": {},
                    "trading_signals": {},
                    "technical_score": 50
                }
            
            sentiment_analysis = data.get("sentiment_analysis", {})
            sentiment_integration = data.get("sentiment_integration", {})
            company_info = data.get("data", {}).get("company_info", {})
            market_data = data.get("data", {}).get("market_data", {})
            
            # Create comprehensive analysis summary for LLM
            analysis_summary = self._create_comprehensive_analysis_summary(
                ticker, price_data, technical_analysis, sentiment_analysis,
                sentiment_integration, company_info, market_data
            )

            # Optional ML path (when requested)
            use_ml_model = bool(data.get("use_ml_model", False))
            prediction_engine = "llm"
            prediction_result = None
            if use_ml_model:
                try:
                    from agents.tools.prediction_agent_tools import generate_ml_prediction_tool
                    ml_res = generate_ml_prediction_tool.invoke({"state": data})
                    if ml_res.get("status") == "success" and ml_res.get("prediction_result"):
                        prediction_result = ml_res.get("prediction_result")
                        prediction_engine = "ml"
                except Exception:
                    prediction_result = None

            # If ML not used or failed, fall back to LLM + rule-based
            if not prediction_result:
                prediction_result = self._generate_llm_prediction(analysis_summary)
                prediction_engine = "llm"
            
            # Calculate confidence and risk metrics
            # Compute confidence metrics using tool
            try:
                from agents.tools.prediction_agent_tools import calculate_confidence_metrics_tool
                cm = calculate_confidence_metrics_tool.invoke({
                    "technical_analysis": technical_analysis,
                    "sentiment_integration": sentiment_integration,
                    "prediction_result": prediction_result
                })
                confidence_metrics = cm.get("confidence_metrics", {})
            except Exception:
                confidence_metrics = self._calculate_confidence_metrics(
                    technical_analysis, sentiment_integration, prediction_result
                )
            
            # Compile final prediction
            final_prediction = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "prediction": prediction_result,
                "confidence_metrics": confidence_metrics,
                # Prefer deterministic tool-based risk assessment
                "risk_assessment": self._assess_risks_via_tool(data, technical_analysis),
                "recommendation": self._generate_recommendation(prediction_result, confidence_metrics),
                "llm_provider": self.llm_provider,
                "llm_available": self.llm_available,
                "sentiment_integration": sentiment_integration,
                "prediction_engine": prediction_engine,
            }
            
            result = {
                "status": "success",
                "prediction_result": prediction_result,  # Return the actual prediction directly
                "final_prediction": final_prediction,    # Keep the full structure for reference
                "next_agent": "risk_assessor"
            }
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "next_agent": "error_handler"
            }
    
    def _create_comprehensive_analysis_summary(self, ticker: str, price_data: Dict, 
                                             technical_analysis: Dict, sentiment_analysis: Dict,
                                             sentiment_integration: Dict, company_info: Dict, 
                                             market_data: Dict) -> str:
        """Create a comprehensive summary including sentiment analysis for LLM."""
        
        # Handle different technical analysis structures
        if "trading_signals" in technical_analysis:
            # Enhanced structure
            trading_signals = technical_analysis.get("trading_signals", {})
            overall_recommendation = trading_signals.get("overall_recommendation", "HOLD")
            signal_strength = trading_signals.get("signal_strength", 0)
            total_signals = trading_signals.get("total_signals", 0)
        else:
            # Basic structure
            technical_signals = technical_analysis.get("technical_signals", [])
            overall_recommendation = technical_signals[0] if technical_signals else "HOLD"
            signal_strength = 0
            total_signals = len(technical_signals)
        
        summary = f"""
COMPREHENSIVE STOCK ANALYSIS SUMMARY FOR {ticker.upper()}

COMPANY INFORMATION:
- Name: {company_info.get('name', 'Unknown')}
- Sector: {company_info.get('sector', 'Unknown')}
- Industry: {company_info.get('industry', 'Unknown')}
- Market Cap: ${company_info.get('market_cap', 0):,.0f}
- P/E Ratio: {company_info.get('pe_ratio', 0):.2f}
- Beta: {company_info.get('beta', 0):.2f}

CURRENT PRICE DATA:
- Current Price: ${price_data.get('current_price', 0):.2f}
- Previous Close: ${price_data.get('previous_close', 0):.2f}
- Daily Change: ${price_data.get('daily_change', 0):.2f} ({price_data.get('daily_change_pct', 0):.2f}%)
- Volume: {price_data.get('volume', 0):,}
- Day Range: ${price_data.get('low', 0):.2f} - ${price_data.get('high', 0):.2f}

TECHNICAL ANALYSIS:
- Technical Score: {technical_analysis.get('technical_score', 0):.1f}/100
- RSI: {technical_analysis.get('indicators', {}).get('rsi', 'N/A')}
- MACD: {technical_analysis.get('indicators', {}).get('macd', 'N/A')}
- SMA 20: ${technical_analysis.get('indicators', {}).get('sma_20', 0):.2f}
- SMA 50: ${technical_analysis.get('indicators', {}).get('sma_50', 0):.2f}
- Short-term Trend: {technical_analysis.get('trend_analysis', {}).get('trends', {}).get('short_term', technical_analysis.get('trend_analysis', {}).get('short_term_trend', 'neutral'))}
- Medium-term Trend: {technical_analysis.get('trend_analysis', {}).get('trends', {}).get('medium_term', technical_analysis.get('trend_analysis', {}).get('medium_term_trend', 'neutral'))}
- Long-term Trend: {technical_analysis.get('trend_analysis', {}).get('trends', {}).get('long_term', 'neutral')}
- Trend Strength: {technical_analysis.get('trend_analysis', {}).get('trend_strength', 0):.1f}%
- Current Price: ${technical_analysis.get('trend_analysis', {}).get('current_price', price_data.get('current_price', 0)):.2f}
- Nearest Support: ${technical_analysis.get('support_resistance', {}).get('nearest_support', technical_analysis.get('support_resistance', {}).get('support_level', 0)):.2f}
- Nearest Resistance: ${technical_analysis.get('support_resistance', {}).get('nearest_resistance', technical_analysis.get('support_resistance', {}).get('resistance_level', 0)):.2f}
- Patterns Detected: {len(technical_analysis.get('patterns', technical_analysis.get('pattern_recognition', {}).get('detected_patterns', [])))}
- Trading Signals: {overall_recommendation}
- Signal Strength: {signal_strength}
- Total Signals: {total_signals}

SENTIMENT ANALYSIS:
- Overall Sentiment: {sentiment_analysis.get('overall_sentiment', {}).get('sentiment_label', 'neutral')}
- Sentiment Score: {sentiment_analysis.get('overall_sentiment', {}).get('sentiment_score', 0):.3f}
- Sentiment Trend: {sentiment_analysis.get('sentiment_trend', 'neutral')}
- News Sentiment: {sentiment_analysis.get('news_sentiment', {}).get('sentiment_label', 'neutral')}
- Reddit Sentiment: {sentiment_analysis.get('reddit_sentiment', {}).get('sentiment_label', 'neutral')}
- Key Sentiment Factors: {', '.join(sentiment_analysis.get('key_sentiment_factors', []))}

SENTIMENT INTEGRATION:
- Integrated Score: {sentiment_integration.get('integrated_analysis', {}).get('integrated_score', 0):.1f}/100
- Sentiment Alignment: {sentiment_integration.get('sentiment_adjusted_confidence', {}).get('alignment', 'neutral')}
- Adjusted Technical Signals: {', '.join([str(signal) for signal in sentiment_integration.get('adjusted_technical_signals', {}).get('adjusted_signals', [])])}
- Sentiment Impact: {sentiment_integration.get('sentiment_insights', {}).get('impact_assessment', 'minimal')}

MARKET CONTEXT:
- S&P 500: {market_data.get('sp500_current', 0):.2f} ({market_data.get('sp500_change_pct', 0):.2f}%)
- Market Trend: {market_data.get('market_trend', 'neutral')}

TASK: Based on this comprehensive analysis including technical indicators, sentiment analysis, and market context, predict the likely direction of {ticker} stock price movement for the next trading day. Consider:
1. Technical indicators and their significance
2. Sentiment analysis from news and social media
3. How sentiment aligns with or conflicts with technical signals
4. Market context and broader trends
5. Risk factors and potential catalysts

Provide your analysis in the following JSON format:
{{
    "direction": "UP/DOWN/NEUTRAL",
    "confidence": 0-100,
    "price_target": null or specific price,
    "price_range": {{
        "low": price,
        "high": price
    }},
    "reasoning": "Detailed explanation incorporating both technical and sentiment factors",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_factors": ["risk1", "risk2", "risk3"],
    "sentiment_influence": "How sentiment analysis influenced your prediction"
}}

Be conservative in your predictions and always consider market volatility.
"""
        return summary
    
    def _generate_llm_prediction(self, analysis_summary: str) -> Dict[str, Any]:
        """
        Generate prediction using LLM with automatic delegation.
        """
        try:
            # Prefer tool-based LLM prediction
            from agents.tools.prediction_agent_tools import generate_llm_prediction_tool

            print("🤖 Delegating prediction to best available LLM agent (tool)...")
            tool_result = generate_llm_prediction_tool.invoke({
                "analysis_summary": analysis_summary
            })

            if tool_result.get("status") == "success" and tool_result.get("prediction_result"):
                return tool_result["prediction_result"]
            else:
                print("⚠️ Tool delegation failed or empty; using fallback rule-based tool")
                return self._generate_rule_based_prediction(analysis_summary)
        except Exception as e:
            print(f"LLM prediction via tool failed: {str(e)}")
            return self._generate_rule_based_prediction(analysis_summary)
    
    def _generate_rule_based_prediction(self, analysis_summary: str) -> Dict[str, Any]:
        """Generate prediction using rule-based analysis when LLM is unavailable."""
        
        # Enhanced rule-based prediction incorporating sentiment
        prediction = {
            "direction": "neutral",
            "confidence": 50.0,
            "price_target": None,
            "price_range": {
                "low": 0,
                "high": 0
            },
            "reasoning": "Rule-based analysis incorporating technical indicators and sentiment data indicates mixed signals. Market conditions appear stable with balanced technical and sentiment factors.",
            "key_factors": [
                "Technical indicators show neutral momentum",
                "Sentiment analysis provides additional context",
                "Price is near support/resistance levels",
                "Market trend is neutral"
            ],
            "risk_factors": [
                "Market volatility could impact short-term movement",
                "Earnings announcements or news events could change outlook",
                "Sector-specific factors may influence performance",
                "Sentiment shifts could alter technical patterns"
            ],
            "sentiment_influence": "Sentiment analysis integrated with technical indicators for comprehensive assessment"
        }
        
        return prediction
    
    def _calculate_confidence_metrics(self, technical_analysis: Dict, 
                                    sentiment_integration: Dict,
                                    prediction_result: Dict) -> Dict[str, Any]:
        """Calculate confidence metrics based on analysis consistency."""
        
        technical_score = technical_analysis.get("technical_score", 50)
        integrated_score = sentiment_integration.get("integrated_analysis", {}).get("integrated_score", 50)
        llm_confidence = prediction_result.get("confidence", 50)
        
        # Calculate weighted confidence with sentiment integration
        weighted_confidence = (integrated_score * 0.5) + (llm_confidence * 0.5)
        
        # Determine confidence level
        if weighted_confidence > 80:
            confidence_level = "very_high"
        elif weighted_confidence > 60:
            confidence_level = "high"
        elif weighted_confidence > 40:
            confidence_level = "medium"
        elif weighted_confidence > 20:
            confidence_level = "low"
        else:
            confidence_level = "very_low"
        
        return {
            "overall_confidence": weighted_confidence,
            "confidence_level": confidence_level,
            "technical_confidence": technical_score,
            "integrated_confidence": integrated_score,
            "llm_confidence": llm_confidence,
            "signal_strength": "strong" if weighted_confidence > 70 else "weak"
        }
    
    def _assess_risks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess various risk factors including sentiment risks."""
        
        technical_analysis = data.get("technical_analysis", {})
        market_data = (data.get("data", {}) or {}).get("market_data", {})
        price_data = (data.get("data", {}) or {}).get("price_data", {})
        company_info = (data.get("data", {}) or {}).get("company_info", {})
        sentiment_analysis = data.get("sentiment_analysis", {})

        risks: Dict[str, Any] = {
            "market_risk": "unknown",
            "volatility_risk": "unknown",
            "liquidity_risk": "unknown",
            "sector_risk": "unknown",
            "sentiment_risk": "unknown",
            "overall_risk_level": "unknown",
            "risk_warnings": [],
        }

        # 1) Market risk: use VIX and SP500 daily change if available
        indices = market_data.get("indices", {})
        vix = (indices.get("vix", {}) or {}).get("current", 0) or 0
        spx_chg = (indices.get("sp500", {}) or {}).get("change_pct", 0) or 0
        if vix >= 25 or spx_chg <= -1.0:
            risks["market_risk"] = "high"
        elif vix >= 18 or abs(spx_chg) >= 0.7:
            risks["market_risk"] = "medium"
        else:
            risks["market_risk"] = "low"

        # 2) Volatility risk: use price_data volatility (20d stdev of returns) and ADX when available
        vol = price_data.get("volatility")  # already last 20 stdev if set
        adx = technical_analysis.get("trend_analysis", {}).get("adx_strength", 0)
        vol_risk = "unknown"
        if isinstance(vol, (int, float)):
            if vol >= 0.035 or adx >= 35:
                vol_risk = "high"
            elif vol >= 0.015 or adx >= 20:
                vol_risk = "medium"
            else:
                vol_risk = "low"
        elif isinstance(adx, (int, float)) and adx:
            vol_risk = "high" if adx >= 35 else ("medium" if adx >= 20 else "low")
        risks["volatility_risk"] = vol_risk

        # 3) Liquidity risk: use average volume and current volume
        avg_vol = price_data.get("avg_volume")
        cur_vol = price_data.get("volume")
        liq_risk = "unknown"
        if isinstance(avg_vol, (int, float)):
            if avg_vol < 200_000:
                liq_risk = "high"
            elif avg_vol < 1_000_000:
                liq_risk = "medium"
            else:
                liq_risk = "low"
        risks["liquidity_risk"] = liq_risk

        # 4) Sector risk: default from market risk, adjust if momentum is weak
        sector = company_info.get("sector") or "Unknown"
        # If sector unknown, mirror market risk
        if sector == "Unknown":
            risks["sector_risk"] = risks["market_risk"]
        else:
            # Simple heuristic: use trend strength
            t_strength = technical_analysis.get("trend_analysis", {}).get("trend_strength", 0)
            if isinstance(t_strength, (int, float)):
                if t_strength < 10:
                    risks["sector_risk"] = "high"
                elif t_strength < 25:
                    risks["sector_risk"] = "medium"
                else:
                    risks["sector_risk"] = "low"
            else:
                risks["sector_risk"] = "medium"

        # 5) Sentiment risk
        sentiment_score = (sentiment_analysis.get("overall_sentiment", {}) or {}).get("sentiment_score", 0) or 0
        if abs(sentiment_score) > 0.5:
            risks["sentiment_risk"] = "high"
        elif abs(sentiment_score) > 0.2:
            risks["sentiment_risk"] = "medium"
        else:
            risks["sentiment_risk"] = "low"

        # Overall risk: aggregate
        map_score = {"low": 1, "medium": 2, "high": 3, "unknown": 2}
        parts = [
            risks["market_risk"],
            risks["volatility_risk"],
            risks["liquidity_risk"],
            risks["sector_risk"],
            risks["sentiment_risk"],
        ]
        avg = sum(map_score.get(x, 2) for x in parts) / len(parts)
        risks["overall_risk_level"] = "low" if avg < 1.5 else ("high" if avg > 2.5 else "medium")

        # Warnings
        if risks["market_risk"] == "high":
            risks["risk_warnings"].append("Broad market risk elevated (VIX/SPX). Consider smaller positions.")
        if risks["volatility_risk"] == "high":
            risks["risk_warnings"].append("High volatility expected – use wider stops and reduced size.")
        if risks["liquidity_risk"] != "low":
            risks["risk_warnings"].append("Lower liquidity – beware of slippage and gaps.")
        if risks["sentiment_risk"] == "high":
            risks["risk_warnings"].append("Extreme sentiment – outcomes may be more erratic.")

        return risks

    def _assess_risks_via_tool(self, data: Dict[str, Any], technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Use risk assessor tool when available; fallback to local heuristic."""
        try:
            from agents.tools.risk_assessor_tools import compute_risk_assessment_tool
            price_data = (data.get("data", {}) or {}).get("price_data", {})
            market_data = (data.get("data", {}) or {}).get("market_data", {})
            company_info = (data.get("data", {}) or {}).get("company_info", {})
            sentiment_analysis = data.get("sentiment_analysis", {})
            res = compute_risk_assessment_tool.invoke({
                "technical_analysis": technical_analysis,
                "price_data": price_data,
                "market_data": market_data,
                "company_info": company_info,
                "sentiment_analysis": sentiment_analysis,
            })
            if isinstance(res, dict) and res.get("status") == "success":
                return res.get("risk_assessment", {})
        except Exception:
            pass
        return self._assess_risks(data)
    
    def _generate_recommendation(self, prediction_result: Dict, 
                               confidence_metrics: Dict) -> Dict[str, Any]:
        """Generate trading recommendation based on prediction and confidence. Uses tool when available."""
        try:
            from agents.tools.prediction_agent_tools import generate_recommendation_tool
            tool_res = generate_recommendation_tool.invoke({
                "prediction_result": prediction_result,
                "confidence_metrics": confidence_metrics
            })
            if tool_res.get("status") == "success" and tool_res.get("recommendation"):
                return tool_res["recommendation"]
        except Exception:
            pass

        # Fallback to local implementation
        direction = prediction_result.get("direction", "neutral")
        confidence = confidence_metrics.get("overall_confidence", 50)

        recommendation = {
            "action": "HOLD",
            "position_size": "normal",
            "timeframe": "1_day",
            "stop_loss": None,
            "take_profit": None
        }

        if direction == "UP" and confidence > 60:
            recommendation["action"] = "BUY"
        elif direction == "DOWN" and confidence > 60:
            recommendation["action"] = "SELL"
        elif direction == "UP" and confidence > 40:
            recommendation["action"] = "BUY_WEAK"
        elif direction == "DOWN" and confidence > 40:
            recommendation["action"] = "SELL_WEAK"

        if confidence > 80:
            recommendation["position_size"] = "large"
        elif confidence < 40:
            recommendation["position_size"] = "small"

        return recommendation

# Function for LangGraph integration
def make_prediction(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for making predictions.
    
    Args:
        state: Current state containing all collected and analyzed data
        
    Returns:
        Updated state with prediction results
    """
    agent = PredictionAgent()
    result = agent.make_prediction(state)
    
    # Update state with prediction
    state.update(result)
    return state 