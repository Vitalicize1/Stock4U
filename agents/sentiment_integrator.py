# agents/sentiment_integrator.py
from typing import Dict, Any
from datetime import datetime
from agents.tools.sentiment_integrator_tools import (
    integrate_sentiment_tool,
)

class SentimentIntegratorAgent:
    """
    Agent responsible for integrating sentiment analysis with technical analysis
    to provide more comprehensive predictions.
    """
    
    def __init__(self):
        self.integration_weights = {
            "technical": 0.6,
            "sentiment": 0.3,
            "market_context": 0.1
        }
    
    def integrate_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate sentiment analysis with existing technical analysis.
        
        Args:
            data: Dictionary containing technical analysis and sentiment data
            
        Returns:
            Dictionary with integrated analysis results
        """
        try:
            # Handle both basic and enhanced technical analysis structures
            technical_analysis = data.get("technical_analysis", {})
            enhanced_technical_analysis = data.get("enhanced_technical_analysis", {})
            sentiment_analysis = data.get("sentiment_analysis", {})
            market_data = data.get("data", {}).get("market_data", {})
            
            # Use enhanced analysis if available, otherwise fall back to basic
            if enhanced_technical_analysis:
                technical_analysis = enhanced_technical_analysis
                print(f"📊 Using enhanced technical analysis for integration")
            elif technical_analysis:
                print(f"📊 Using basic technical analysis for integration")
            else:
                return {
                    "status": "error",
                    "error": "No technical analysis available for integration",
                    "next_agent": "prediction_agent"
                }
            
            # Prefer deterministic tool orchestration
            try:
                tool_res = integrate_sentiment_tool.invoke({
                    "technical_analysis": technical_analysis,
                    "sentiment_analysis": sentiment_analysis,
                    "market_data": market_data,
                })
            except Exception:
                tool_res = None

            if isinstance(tool_res, dict) and tool_res.get("status") == "success":
                integration_results = tool_res.get("sentiment_integration", {})
                integration_results["integration_quality"] = self._assess_integration_quality(technical_analysis, sentiment_analysis)
                # Wire adjusted signals back into technical analysis output path
                try:
                    adj = (integration_results.get("adjusted_technical_signals") or {})
                    if adj:
                        # Select which TA block to mutate
                        ta_key = "enhanced_technical_analysis" if data.get("enhanced_technical_analysis") else "technical_analysis"
                        ta = data.get(ta_key) or {}
                        ts = (ta.get("trading_signals") or {}) if isinstance(ta, dict) else {}
                        # Apply adjusted recommendation and optionally tweak strength
                        if ts:
                            original_strength = float(ts.get("signal_strength", 0) or 0)
                            sentiment_adj = float(adj.get("sentiment_adjustment", 0) or 0)
                            ts["signals"] = adj.get("adjusted_signals", ts.get("signals", []))
                            ts["overall_recommendation"] = adj.get("adjusted_recommendation", ts.get("overall_recommendation", "HOLD"))
                            # Smooth numeric bias without double counting (cap +/-1)
                            ts["signal_strength"] = original_strength + max(-1.0, min(1.0, sentiment_adj))
                            # Write back into state
                            ta["trading_signals"] = ts
                            data[ta_key] = ta
                except Exception:
                    pass
            else:
                # Fallback to local implementation
                integrated_analysis = self._combine_analyses(technical_analysis, sentiment_analysis, market_data)
                adjusted_signals = self._adjust_signals_with_sentiment(technical_analysis, sentiment_analysis)
                sentiment_adjusted_confidence = self._calculate_sentiment_adjusted_confidence(
                    technical_analysis, sentiment_analysis
                )
                sentiment_insights = self._generate_sentiment_insights(sentiment_analysis, technical_analysis)
                integration_results = {
                    "integrated_analysis": integrated_analysis,
                    "adjusted_technical_signals": adjusted_signals,
                    "sentiment_adjusted_confidence": sentiment_adjusted_confidence,
                    "sentiment_insights": sentiment_insights,
                    "integration_quality": self._assess_integration_quality(technical_analysis, sentiment_analysis)
                }
                # Wire adjusted signals back into technical analysis output path (fallback path)
                try:
                    adj = integration_results.get("adjusted_technical_signals") or {}
                    if adj:
                        ta_key = "enhanced_technical_analysis" if data.get("enhanced_technical_analysis") else "technical_analysis"
                        ta = data.get(ta_key) or {}
                        ts = (ta.get("trading_signals") or {}) if isinstance(ta, dict) else {}
                        if ts:
                            original_strength = float(ts.get("signal_strength", 0) or 0)
                            sentiment_adj = float(adj.get("sentiment_adjustment", 0) or 0)
                            ts["signals"] = adj.get("adjusted_signals", ts.get("signals", []))
                            ts["overall_recommendation"] = adj.get("adjusted_recommendation", ts.get("overall_recommendation", "HOLD"))
                            ts["signal_strength"] = original_strength + max(-1.0, min(1.0, sentiment_adj))
                            ta["trading_signals"] = ts
                            data[ta_key] = ta
                except Exception:
                    pass
            
            # Best-effort: append daily sentiment sample for ML training
            try:
                from ml.sentiment_logger import append_sentiment_sample
                tkr = data.get("ticker", "").upper()
                ts = (tool_res or {}).get("timestamp") or datetime.now().isoformat()
                s_score = (sentiment_analysis.get("overall_sentiment", {}) or {}).get("sentiment_score", 0.0)
                append_sentiment_sample(tkr, ts, float(s_score or 0.0))
            except Exception:
                pass

            return {
                "status": "success",
                "sentiment_integration": integration_results,
                "next_agent": "prediction_agent"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Sentiment integration failed: {str(e)}",
                "next_agent": "prediction_agent"
            }
    
    def _combine_analyses(self, technical_analysis: Dict, sentiment_analysis: Dict, market_data: Dict) -> Dict[str, Any]:
        """Combine technical and sentiment analyses."""
        
        # Handle different technical analysis structures
        if "technical_score" in technical_analysis:
            technical_score = technical_analysis.get("technical_score", 50)
        elif "trading_signals" in technical_analysis:
            # Enhanced structure - calculate score from signals
            signals = technical_analysis.get("trading_signals", {})
            signal_strength = signals.get("signal_strength", 0)
            technical_score = 50 + (signal_strength * 10)  # Convert to 0-100 scale
        else:
            technical_score = 50  # Default score
        
        sentiment_score = sentiment_analysis.get("overall_sentiment", {}).get("sentiment_score", 0)
        
        # Convert sentiment score (-1 to 1) to 0-100 scale
        sentiment_normalized = (sentiment_score + 1) * 50
        
        # Market context score (simplified)
        market_trend = market_data.get("market_trend", "neutral")
        market_score = 70 if market_trend == "bullish" else 30 if market_trend == "bearish" else 50
        
        # Weighted combination
        integrated_score = (
            technical_score * self.integration_weights["technical"] +
            sentiment_normalized * self.integration_weights["sentiment"] +
            market_score * self.integration_weights["market_context"]
        )
        
        return {
            "integrated_score": integrated_score,
            "technical_contribution": technical_score * self.integration_weights["technical"],
            "sentiment_contribution": sentiment_normalized * self.integration_weights["sentiment"],
            "market_contribution": market_score * self.integration_weights["market_context"],
            "integration_breakdown": {
                "technical_weight": self.integration_weights["technical"],
                "sentiment_weight": self.integration_weights["sentiment"],
                "market_weight": self.integration_weights["market_context"]
            }
        }
    
    def _adjust_signals_with_sentiment(self, technical_analysis: Dict, sentiment_analysis: Dict) -> Dict[str, Any]:
        """Adjust technical signals based on sentiment analysis."""
        
        # Get technical signals
        if "trading_signals" in technical_analysis:
            # Enhanced structure
            signals = technical_analysis.get("trading_signals", {})
            original_signals = signals.get("signals", [])
            overall_recommendation = signals.get("overall_recommendation", "HOLD")
        else:
            # Basic structure
            original_signals = technical_analysis.get("technical_signals", [])
            overall_recommendation = "HOLD" if not original_signals else original_signals[0]
        
        # Get sentiment score
        sentiment_score = sentiment_analysis.get("overall_sentiment", {}).get("sentiment_score", 0)
        sentiment_label = sentiment_analysis.get("overall_sentiment", {}).get("sentiment_label", "neutral")
        
        # Adjust signals based on sentiment
        adjusted_signals = []
        sentiment_adjustment = 0
        
        # Strong positive sentiment strengthens buy signals
        if sentiment_score > 0.3:
            sentiment_adjustment = 1
            for signal in original_signals:
                if isinstance(signal, dict) and signal.get("type") == "BUY":
                    signal["strength"] = "strong"
                    signal["sentiment_boost"] = True
                adjusted_signals.append(signal)
        
        # Strong negative sentiment strengthens sell signals
        elif sentiment_score < -0.3:
            sentiment_adjustment = -1
            for signal in original_signals:
                if isinstance(signal, dict) and signal.get("type") == "SELL":
                    signal["strength"] = "strong"
                    signal["sentiment_boost"] = True
                adjusted_signals.append(signal)
        
        # Neutral sentiment - no adjustment
        else:
            adjusted_signals = original_signals
        
        # Determine adjusted overall recommendation
        if sentiment_adjustment > 0 and "BUY" in overall_recommendation:
            adjusted_recommendation = "STRONG_BUY" if overall_recommendation == "BUY" else overall_recommendation
        elif sentiment_adjustment < 0 and "SELL" in overall_recommendation:
            adjusted_recommendation = "STRONG_SELL" if overall_recommendation == "SELL" else overall_recommendation
        else:
            adjusted_recommendation = overall_recommendation
        
        return {
            "original_signals": original_signals,
            "adjusted_signals": adjusted_signals,
            "original_recommendation": overall_recommendation,
            "adjusted_recommendation": adjusted_recommendation,
            "sentiment_adjustment": sentiment_adjustment,
            "sentiment_label": sentiment_label
        }
    
    def _calculate_sentiment_adjusted_confidence(self, technical_analysis: Dict, sentiment_analysis: Dict) -> Dict[str, Any]:
        """Calculate confidence adjusted by sentiment."""
        
        # Get technical confidence
        if "trading_signals" in technical_analysis:
            # Enhanced structure
            signals = technical_analysis.get("trading_signals", {})
            technical_confidence = signals.get("signal_strength", 0)
        else:
            # Basic structure
            technical_confidence = technical_analysis.get("technical_score", 50)
        
        # Get sentiment confidence
        sentiment_confidence = sentiment_analysis.get("overall_sentiment", {}).get("confidence", 0.5)
        
        # Calculate alignment
        alignment = self._check_sentiment_alignment(technical_analysis, sentiment_analysis)
        
        # Adjust confidence based on alignment
        if alignment == "aligned":
            adjusted_confidence = min(100, technical_confidence * 1.2)
        elif alignment == "conflicting":
            adjusted_confidence = technical_confidence * 0.8
        else:
            adjusted_confidence = technical_confidence
        
        return {
            "technical_confidence": technical_confidence,
            "sentiment_confidence": sentiment_confidence,
            "adjusted_confidence": adjusted_confidence,
            "alignment": alignment,
            "confidence_boost": adjusted_confidence - technical_confidence
        }
    
    def _check_sentiment_alignment(self, technical_analysis: Dict, sentiment_analysis: Dict) -> str:
        """Check if technical and sentiment signals are aligned."""
        
        # Get technical direction
        if "trading_signals" in technical_analysis:
            signals = technical_analysis.get("trading_signals", {})
            technical_direction = signals.get("overall_recommendation", "HOLD")
        else:
            technical_signals = technical_analysis.get("technical_signals", [])
            technical_direction = "HOLD" if not technical_signals else technical_signals[0]
        
        # Get sentiment direction
        sentiment_score = sentiment_analysis.get("overall_sentiment", {}).get("sentiment_score", 0)
        sentiment_direction = "BUY" if sentiment_score > 0.1 else "SELL" if sentiment_score < -0.1 else "HOLD"
        
        # Check alignment
        if "BUY" in technical_direction and sentiment_direction == "BUY":
            return "aligned"
        elif "SELL" in technical_direction and sentiment_direction == "SELL":
            return "aligned"
        elif "HOLD" in technical_direction and sentiment_direction == "HOLD":
            return "neutral"
        else:
            return "conflicting"
    
    def _generate_sentiment_insights(self, sentiment_analysis: Dict, technical_analysis: Dict) -> Dict[str, Any]:
        """Generate insights from sentiment analysis."""
        
        sentiment_score = sentiment_analysis.get("overall_sentiment", {}).get("sentiment_score", 0)
        sentiment_label = sentiment_analysis.get("overall_sentiment", {}).get("sentiment_label", "neutral")
        
        # Analyze sentiment impact
        impact = self._assess_sentiment_impact(sentiment_score, technical_analysis)
        
        # Generate recommendations
        recommendations = self._generate_sentiment_recommendations(sentiment_analysis)
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "impact_assessment": impact,
            "recommendations": recommendations,
            "key_insights": [
                f"Market sentiment is {sentiment_label}",
                f"Sentiment impact: {impact}",
                f"Recommendations: {len(recommendations)} actionable items"
            ]
        }
    
    def _assess_sentiment_impact(self, sentiment_score: float, technical_analysis: Dict) -> str:
        """Assess the impact of sentiment on technical analysis."""
        
        if sentiment_score > 0.5:
            return "strong_positive"
        elif sentiment_score > 0.1:
            return "moderate_positive"
        elif sentiment_score < -0.5:
            return "strong_negative"
        elif sentiment_score < -0.1:
            return "moderate_negative"
        else:
            return "neutral"
    
    def _generate_sentiment_recommendations(self, sentiment_analysis: Dict) -> list:
        """Generate recommendations based on sentiment analysis."""
        
        recommendations = []
        sentiment_score = sentiment_analysis.get("overall_sentiment", {}).get("sentiment_score", 0)
        
        if sentiment_score > 0.3:
            recommendations.extend([
                "Consider increasing position size due to positive sentiment",
                "Monitor for potential momentum continuation",
                "Watch for sentiment-driven breakouts"
            ])
        elif sentiment_score < -0.3:
            recommendations.extend([
                "Consider reducing position size due to negative sentiment",
                "Monitor for potential sentiment-driven reversals",
                "Watch for support levels as sentiment improves"
            ])
        else:
            recommendations.extend([
                "Sentiment is neutral - focus on technical analysis",
                "Monitor for sentiment shifts",
                "Maintain current position sizing"
            ])
        
        return recommendations
    
    def _assess_integration_quality(self, technical_analysis: Dict, sentiment_analysis: Dict) -> Dict[str, Any]:
        """Assess the quality of the integration."""
        
        # Check data completeness
        technical_complete = bool(technical_analysis)
        sentiment_complete = bool(sentiment_analysis)
        
        # Check alignment
        alignment = self._check_sentiment_alignment(technical_analysis, sentiment_analysis)
        
        # Calculate quality score
        quality_score = 0
        if technical_complete:
            quality_score += 50
        if sentiment_complete:
            quality_score += 30
        if alignment == "aligned":
            quality_score += 20
        elif alignment == "neutral":
            quality_score += 10
        
        return {
            "quality_score": quality_score,
            "technical_complete": technical_complete,
            "sentiment_complete": sentiment_complete,
            "alignment": alignment,
            "integration_quality": "high" if quality_score >= 80 else "medium" if quality_score >= 60 else "low"
        }

# Function for LangGraph integration
def integrate_sentiment(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for sentiment integration.
    
    Args:
        state: Current state containing technical and sentiment analysis
        
    Returns:
        Updated state with integrated analysis
    """
    print(f"🔗 Sentiment integrator starting integration...")
    
    agent = SentimentIntegratorAgent()
    result = agent.integrate_sentiment(state)
    
    # Update state with sentiment integration
    state.update(result)
    return state 