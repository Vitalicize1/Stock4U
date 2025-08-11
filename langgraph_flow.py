from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from typing import TypedDict, Dict, Any, Optional
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from agents import (
    data_collector, 
    technical_analyzer, 
    sentiment_analyzer,
    sentiment_integrator,
    prediction_agent,
    orchestrator, 
    evaluator_optimizer, 
    elicitation,
    chatbot_agent
)

# Import tools for each agent
from agents.tools.data_collector_tools import (
    collect_price_data,
    collect_company_info,
    collect_market_data,
    calculate_technical_indicators,
    validate_data_quality,
    collect_comprehensive_data
)

from agents.tools.technical_analyzer_tools import (
    calculate_advanced_indicators,
    identify_chart_patterns,
    analyze_support_resistance,
    perform_trend_analysis,
    generate_trading_signals,
    validate_technical_analysis,
    calculate_short_term_indicators,
    compute_supertrend,
    compute_ichimoku_cloud,
    compute_keltner_channels,
    compute_donchian_channels,
)

from agents.tools.orchestrator_tools import (
    validate_ticker_symbol,
    check_market_status,
    determine_analysis_parameters,
    initialize_workflow_state,
    coordinate_workflow_stage,
    handle_workflow_error,
    delegate_prediction_agent
)

# Sentiment tools
from agents.tools.sentiment_analyzer_tools import (
    fetch_news_articles,
    fetch_reddit_posts,
    analyze_text_sentiment,
    aggregate_sentiment,
    compute_overall_sentiment,
)
from agents.tools.sentiment_integrator_tools import (
    integrate_sentiment_tool,
    integrate_scores,
    adjust_signals_with_sentiment,
    calculate_alignment_and_confidence,
    generate_sentiment_insights_tool,
)
# Prediction tools
from agents.tools.prediction_agent_tools import (
    generate_llm_prediction_tool,
    generate_rule_based_prediction_tool,
    generate_ml_prediction_tool,
    calculate_confidence_metrics_tool,
    generate_recommendation_tool,
)
from agents.tools.evaluator_optimizer_tools import (
    evaluate_overall_tool,
    assess_prediction_quality_tool,
    assess_technical_consistency_tool,
    assess_risk_adequacy_tool,
    assess_recommendation_strength_tool,
    calculate_evaluation_score_tool,
    generate_optimization_suggestions_tool,
)
from agents.tools.elicitation_tools import (
    elicit_confirmation_tool,
    assemble_final_summary_tool,
    format_final_summary_text_tool,
)

# Import LLM clients
from llm.gemini_client import get_gemini_client
from llm.groq_client import get_groq_client
from utils.result_cache import get_cached_result, set_cached_result
from functools import lru_cache

# Define the state schema for LangGraph
class AgentState(TypedDict):
    ticker: str
    timeframe: str
    timestamp: Optional[str]
    status: str
    workflow_start_time: Optional[str]
    pipeline_stage: Optional[str]
    next_agent: Optional[str]
    data: Optional[Dict[str, Any]]
    technical_analysis: Optional[Dict[str, Any]]
    enhanced_technical_analysis: Optional[Dict[str, Any]]
    sentiment_analysis: Optional[Dict[str, Any]]
    sentiment_integration: Optional[Dict[str, Any]]
    prediction_result: Optional[Dict[str, Any]]
    confidence_metrics: Optional[Dict[str, Any]]
    recommendation: Optional[Dict[str, Any]]
    evaluation_results: Optional[Dict[str, Any]]
    final_summary: Optional[Dict[str, Any]]
    error: Optional[str]
    # Chatbot-specific fields
    user_query: Optional[str]
    chatbot_response: Optional[Dict[str, Any]]
    # Orchestrator tools enhanced fields
    company_name: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    market_cap: Optional[float]
    currency: Optional[str]
    market_status: Optional[str]
    is_market_open: Optional[bool]
    analysis_depth: Optional[str]
    analysis_parameters: Optional[Dict[str, Any]]
    recommended_indicators: Optional[list]
    data_sources: Optional[list]
    workflow_stages: Optional[list]
    current_stage: Optional[int]
    total_stages: Optional[int]
    workflow_version: Optional[str]
    prediction_type: Optional[str]
    orchestrator_tools_used: Optional[list]
    validation_results: Optional[Dict[str, Any]]
    market_status_info: Optional[Dict[str, Any]]
    # Messages for proper LangGraph tool integration
    messages: Optional[list]

def create_llm_agent_wrapper(agent, agent_name: str):
    """
    Create a wrapper for LLM agents that properly handles state updates.
    
    Args:
        agent: The create_react_agent instance
        agent_name: Name of the agent for logging
        
    Returns:
        Wrapped agent function that updates state properly
    """
    def wrapper(state):
        try:
            print(f"🤖 {agent_name} starting...")

            # For prediction_agent, deterministically call tools to avoid provider tool-call quirks
            if agent_name == "prediction_agent":
                try:
                    from agents.prediction_agent import PredictionAgent
                    # Prefer enhanced analysis
                    technical_analysis = state.get("enhanced_technical_analysis") or state.get("technical_analysis", {})
                    price_data = (state.get("data", {}) or {}).get("price_data", {})
                    company_info = (state.get("data", {}) or {}).get("company_info", {})
                    market_data = (state.get("data", {}) or {}).get("market_data", {})
                    sentiment_analysis = state.get("sentiment_analysis", {})
                    sentiment_integration = state.get("sentiment_integration", {})

                    # Build analysis summary using existing implementation
                    pa = PredictionAgent()
                    analysis_summary = pa._create_comprehensive_analysis_summary(
                        state.get("ticker", "UNKNOWN"),
                        price_data,
                        technical_analysis,
                        sentiment_analysis,
                        sentiment_integration,
                        company_info,
                        market_data,
                    )

                    # Respect low_api_mode flag to optionally skip LLM
                    low_api_mode = bool(state.get("low_api_mode", False))
                    # Optional ML mode: prefer ML over LLM if flag is set
                    use_ml = bool(state.get("use_ml_model", False))
                    if use_ml:
                        pred_res = generate_ml_prediction_tool.invoke({
                            "state": state
                        })
                        if pred_res.get("status") != "success" or not pred_res.get("prediction_result"):
                            # fallback to LLM or rule-based depending on low_api_mode
                            if not low_api_mode:
                                pred_res = generate_llm_prediction_tool.invoke({
                                    "analysis_summary": analysis_summary
                                })
                            else:
                                pred_res = generate_rule_based_prediction_tool.invoke({
                                    "analysis_summary": analysis_summary
                                })
                    else:
                        if not low_api_mode:
                            pred_res = generate_llm_prediction_tool.invoke({
                                "analysis_summary": analysis_summary
                            })
                            if pred_res.get("status") != "success" or not pred_res.get("prediction_result"):
                                pred_res = generate_rule_based_prediction_tool.invoke({
                                    "analysis_summary": analysis_summary
                                })
                        else:
                            pred_res = generate_rule_based_prediction_tool.invoke({
                                "analysis_summary": analysis_summary
                            })
                    prediction = pred_res.get("prediction_result", {})

                    conf_res = calculate_confidence_metrics_tool.invoke({
                        "technical_analysis": technical_analysis,
                        "sentiment_integration": sentiment_integration,
                        "prediction_result": prediction,
                        "sentiment_analysis": state.get("sentiment_analysis", {}),
                    })
                    confidence_metrics = conf_res.get("confidence_metrics", {})

                    rec_res = generate_recommendation_tool.invoke({
                        "prediction_result": prediction,
                        "confidence_metrics": confidence_metrics,
                    })
                    recommendation = rec_res.get("recommendation")

                    # Embed confidence and recommendation inside prediction_result as well
                    if isinstance(prediction, dict):
                        prediction["confidence_metrics"] = confidence_metrics
                        if isinstance(recommendation, dict):
                            prediction["recommendation"] = recommendation

                    state.update({
                        "prediction_result": prediction,
                        "confidence_metrics": confidence_metrics,
                        "recommendation": recommendation,
                        "status": "success",
                        "next_agent": get_next_agent(agent_name),
                    })
                    # Also mirror into top-level result shape expected by downstream nodes
                    result_payload = {
                        "prediction_result": prediction,
                        "confidence_metrics": confidence_metrics,
                        "recommendation": recommendation,
                        "status": "success",
                        "next_agent": get_next_agent(agent_name),
                    }
                    return {**state, **result_payload}
                    print("✅ prediction_agent completed successfully via deterministic tools")
                    
                except Exception as e:
                    print(f"❌ prediction_agent deterministic tools failed: {e}")
                    state["status"] = "error"
                    state["error"] = f"prediction_agent failed: {e}"
                    return state

            # For technical_analyzer, deterministically call tools to avoid slow LLM planning
            if agent_name == "technical_analyzer":
                try:
                    ticker = state.get("ticker")
                    timeframe = state.get("timeframe", "1d")
                    # Use 3 months for daily timeframe by default
                    period = "3mo" if timeframe in ("1d", "1w", "1mo") else "6mo"
                    fast_ta = bool(state.get("fast_ta_mode", False))

                    # Call tools directly
                    adv = calculate_advanced_indicators.invoke({"ticker": ticker, "period": period})
                    # In fast TA mode, skip heavy tools and rely on signals only
                    if fast_ta:
                        trend = {"status": "success", "trend_analysis": {}}
                    else:
                        trend = perform_trend_analysis.invoke({"ticker": ticker, "period": period})
                    signals = generate_trading_signals.invoke({"ticker": ticker, "period": period})

                    # Optional short-term and confirmation tools (best-effort)
                    short_term = {}
                    ichimoku = {}
                    supertrend = {}
                    keltner = {}
                    donchian = {}
                    if not fast_ta:
                        try:
                            short_term = calculate_short_term_indicators.invoke({"ticker": ticker, "period": period})
                        except Exception:
                            pass
                        try:
                            ichimoku = compute_ichimoku_cloud.invoke({"ticker": ticker, "period": period})
                        except Exception:
                            pass
                        try:
                            supertrend = compute_supertrend.invoke({"ticker": ticker, "period": period})
                        except Exception:
                            pass
                        try:
                            keltner = compute_keltner_channels.invoke({"ticker": ticker, "period": period})
                        except Exception:
                            pass
                        try:
                            donchian = compute_donchian_channels.invoke({"ticker": ticker, "period": period})
                        except Exception:
                            pass

                    # Merge indicators
                    indicators = {}
                    for src in [
                        (adv or {}).get("current_indicators"),
                        (short_term or {}).get("current_indicators"),
                        (ichimoku or {}).get("current_indicators"),
                        (supertrend or {}).get("current_indicators"),
                        (keltner or {}).get("current_indicators"),
                        (donchian or {}).get("current_indicators"),
                    ]:
                        if isinstance(src, dict):
                            indicators.update(src)

                    # Ensure core keys for UI are present; compute fallbacks if needed
                    core_keys = ["rsi", "macd", "sma_20", "sma_50", "sma_200"]
                    needs_fallback = any(indicators.get(k) is None for k in core_keys) or not all(k in indicators for k in core_keys)
                    if needs_fallback:
                        try:
                            from agents.tools.technical_analyzer_tools import TechnicalAnalyzerTools
                            _tools = TechnicalAnalyzerTools()
                            _df = _tools._get_price_data(ticker, period)
                            if _df is not None and not _df.empty:
                                _ind = _tools._calculate_basic_indicators(_df)
                                def _last(series):
                                    try:
                                        val = series.iloc[-1]
                                        return float(val) if val == val else None  # NaN check
                                    except Exception:
                                        return None
                                indicators.setdefault("rsi", _last(_ind.get("rsi")))
                                indicators.setdefault("macd", _last(_ind.get("macd")))
                                indicators.setdefault("sma_20", _last(_ind.get("sma_20")))
                                indicators.setdefault("sma_50", _last(_ind.get("sma_50")))
                                indicators.setdefault("sma_200", _last(_ind.get("sma_200")))
                        except Exception:
                            pass

                    trend_analysis = trend if isinstance(trend, dict) else {}
                    if "status" in trend_analysis:
                        # unwrap if tool returned structured payload
                        trend_analysis = {
                            k: v for k, v in trend_analysis.items() if k not in ("status", "ticker")
                        }

                    trading_signals = signals if isinstance(signals, dict) else {}
                    if trading_signals.get("status") == "success":
                        # keep as is
                        pass

                    # Compute technical score (lightweight)
                    s_strength = float(trading_signals.get("signal_strength", 0) or 0)
                    t_strength = float(trend_analysis.get("trend_strength", 0) or 0)
                    tech_score = max(0.0, min(100.0, 50.0 + s_strength * 8.0 + (t_strength - 50.0) * 0.2))

                    technical_analysis = {
                        "indicators": indicators,
                        "trend_analysis": trend_analysis,
                        "trading_signals": trading_signals,
                        "technical_score": tech_score,
                    }

                    state.update({
                        "status": "success",
                        "technical_analysis": technical_analysis,
                        "next_agent": get_next_agent(agent_name),
                    })
                    return state

                except Exception as e:
                    print(f"❌ technical_analyzer deterministic tools failed: {e}")
                    state["status"] = "error"
                    state["error"] = f"technical_analyzer failed: {e}"
                    return state
            
            # Prepare the state for the agent with proper input format
            # For prediction_agent, include rich context so tools can be called with real data
            if agent_name == "prediction_agent":
                context_payload = {
                    "ticker": state.get("ticker"),
                    "timeframe": state.get("timeframe"),
                    "price_data": (state.get("data", {}) or {}).get("price_data", {}),
                    "company_info": (state.get("data", {}) or {}).get("company_info", {}),
                    "market_data": (state.get("data", {}) or {}).get("market_data", {}),
                    "technical_analysis": state.get("enhanced_technical_analysis") or state.get("technical_analysis", {}),
                    "sentiment_analysis": state.get("sentiment_analysis", {}),
                    "sentiment_integration": state.get("sentiment_integration", {}),
                }
                import json as _json
                prediction_input = (
                    "You are provided the complete analysis context as JSON. "
                    "1) Build an analysis_summary string from it and call generate_llm_prediction_tool. "
                    "2) Then call calculate_confidence_metrics_tool using the given technical_analysis, sentiment_integration, and your prediction_result. "
                    "3) Finally call generate_recommendation_tool. "
                    "Context JSON follows:\n" + _json.dumps(context_payload)
                )
                agent_state = {
                    "messages": state.get("messages", []),
                    "ticker": state.get("ticker"),
                    "timeframe": state.get("timeframe"),
                    "status": state.get("status", "initialized"),
                    "input": prediction_input,
                }
            else:
                agent_state = {
                    "messages": state.get("messages", []),
                    "ticker": state.get("ticker"),
                    "timeframe": state.get("timeframe"),
                    "status": state.get("status", "initialized"),
                    "input": state.get("input", f"Process {state.get('ticker', 'unknown')} with {state.get('timeframe', '1d')} timeframe")
                }
            
            # Run the LLM agent
            result = agent.invoke(agent_state)
            
            # Extract the final message from the agent
            if result.get("messages"):
                final_message = result["messages"][-1]
                
                # If it's an AIMessage with tool calls, the agent is still working
                if hasattr(final_message, 'tool_calls') and final_message.tool_calls:
                    print(f"🔄 {agent_name} has tool calls - continuing...")
                    return result
                
                # If it's a final AIMessage without tool calls, the agent is done
                if hasattr(final_message, 'content') and final_message.content:
                    print(f"✅ {agent_name} completed successfully")
                    
                    # Extract tool results and update state based on agent type
                    if agent_name == "orchestrator":
                        # Extract orchestrator results from tool messages
                        for msg in result["messages"]:
                            if hasattr(msg, 'name') and msg.name == "initialize_workflow_state":
                                try:
                                    import json
                                    workflow_data = json.loads(msg.content)
                                    result.update(workflow_data)
                                except:
                                    pass
                    
                    elif agent_name == "data_collector":
                        # Extract data collection results
                        data_results = {}
                        for msg in result["messages"]:
                            if hasattr(msg, 'name') and msg.name == "collect_comprehensive_data":
                                try:
                                    # Try to parse as JSON first
                                    if isinstance(msg.content, str):
                                        import json
                                        tool_data = json.loads(msg.content)
                                        data_results.update(tool_data)
                                        print(f"✅ Extracted data from {msg.name}: {len(tool_data)} fields")
                                    elif isinstance(msg.content, dict):
                                        data_results.update(msg.content)
                                        print(f"✅ Extracted data from {msg.name}: {len(msg.content)} fields")
                                except Exception as e:
                                    print(f"⚠️ Failed to parse {msg.name} result: {e}")
                                    # Try to parse as dict if it's already a dict
                                    if isinstance(msg.content, dict):
                                        data_results.update(msg.content)
                                        print(f"✅ Used dict content from {msg.name}")
                                    elif isinstance(msg.content, str):
                                        # Try to evaluate as literal
                                        try:
                                            import ast
                                            tool_data = ast.literal_eval(msg.content)
                                            data_results.update(tool_data)
                                            print(f"✅ Used ast.literal_eval for {msg.name}")
                                        except:
                                            print(f"⚠️ Could not parse {msg.name} content: {msg.content[:100]}...")
                        
                        if data_results:
                            result["data"] = data_results
                            result["status"] = "success"
                            result["next_agent"] = get_next_agent(agent_name)
                            print(f"✅ Data collector completed with {len(data_results)} data fields")
                        else:
                            print("⚠️ No data extracted, using fallback")
                            # Fallback: use traditional data collector
                            from agents.data_collector import collect_data
                            data_result = collect_data(state)
                            result.update(data_result)
                    
                    elif agent_name == "technical_analyzer":
                        # Extract technical analysis results and normalize schema for the dashboard
                        indicators_res = {}
                        trend_res = {}
                        signals_res = {}
                        short_term_res = {}
                        ichimoku_res = {}
                        supertrend_res = {}
                        keltner_res = {}
                        donchian_res = {}
                        try:
                            import json as _json
                        except Exception:
                            _json = None
                        try:
                            import ast as _ast
                        except Exception:
                            _ast = None

                        for msg in result.get("messages", []):
                            if hasattr(msg, 'name') and msg.name in [
                                "calculate_advanced_indicators",
                                "perform_trend_analysis",
                                "generate_trading_signals",
                                "calculate_short_term_indicators",
                                "compute_supertrend",
                                "compute_ichimoku_cloud",
                                "compute_keltner_channels",
                                "compute_donchian_channels",
                            ]:
                                payload = None
                                try:
                                    payload = _json.loads(msg.content) if _json and isinstance(msg.content, str) else msg.content
                                except Exception:
                                    # Try ast.literal_eval for python-literal strings
                                    try:
                                        payload = _ast.literal_eval(msg.content) if _ast and isinstance(msg.content, str) else None
                                    except Exception:
                                        payload = msg.content if isinstance(msg.content, dict) else None

                                if not isinstance(payload, dict):
                                    continue

                                if msg.name == "calculate_advanced_indicators":
                                    # Prefer nested current_indicators/data if present
                                    indicators_res = (
                                        payload.get("current_indicators")
                                        or payload.get("indicators")
                                        or payload.get("data")
                                        or payload
                                    )
                                elif msg.name == "perform_trend_analysis":
                                    trend_res = (
                                        payload.get("trend_analysis")
                                        or payload.get("data")
                                        or payload
                                    )
                                elif msg.name == "generate_trading_signals":
                                    # Keep flat structure; dashboard expects nested under trading_signals
                                    signals_res = (
                                        payload.get("trading_signals")
                                        or payload.get("data")
                                        or payload
                                    )
                                elif msg.name == "calculate_short_term_indicators":
                                    short_term_res = (
                                        payload.get("current_indicators")
                                        or payload.get("data")
                                        or payload
                                    )
                                elif msg.name == "compute_supertrend":
                                    supertrend_res = (
                                        payload.get("current_indicators")
                                        or payload.get("data")
                                        or payload
                                    )
                                elif msg.name == "compute_ichimoku_cloud":
                                    ichimoku_res = (
                                        payload.get("current_indicators")
                                        or payload.get("data")
                                        or payload
                                    )
                                elif msg.name == "compute_keltner_channels":
                                    keltner_res = (
                                        payload.get("current_indicators")
                                        or payload.get("data")
                                        or payload
                                    )
                                elif msg.name == "compute_donchian_channels":
                                    donchian_res = (
                                        payload.get("current_indicators")
                                        or payload.get("data")
                                        or payload
                                    )

                        # If we captured anything, build the normalized technical_analysis
                        if indicators_res or trend_res or signals_res or short_term_res or ichimoku_res or supertrend_res or keltner_res or donchian_res:
                            merged_indicators = {}
                            for src in [indicators_res, short_term_res, ichimoku_res, supertrend_res, keltner_res, donchian_res]:
                                if isinstance(src, dict):
                                    merged_indicators.update(src)
                            normalized = {
                                "indicators": merged_indicators,
                                "trend_analysis": trend_res or {},
                                "trading_signals": signals_res or {},
                            }

                            # Compute a lightweight technical_score if not provided
                            if "technical_score" not in normalized:
                                base_score = 50.0
                                try:
                                    s_strength = float((signals_res or {}).get("signal_strength", 0))
                                except Exception:
                                    s_strength = 0.0
                                try:
                                    t_strength = float((trend_res or {}).get("trend_strength", 0))
                                except Exception:
                                    t_strength = 0.0

                                score = base_score + (s_strength * 8.0) + ((t_strength - 50.0) * 0.2)
                                # Clamp to [0, 100]
                                score = max(0.0, min(100.0, score))
                                normalized["technical_score"] = score

                            result["technical_analysis"] = normalized
                            result["status"] = "success"
                            result["next_agent"] = get_next_agent(agent_name)
                        else:
                            # Fallback: use traditional technical analyzer
                            from agents.technical_analyzer import analyze_technical
                            tech_result = analyze_technical(state)
                            result.update(tech_result)
                    
                    # Update the state with the result
                    if agent_name == "prediction_agent":
                        # Collect tool outputs
                        pred = None
                        conf = None
                        rec = None
                        import json
                        for msg in result.get("messages", []):
                            if hasattr(msg, 'name') and msg.name == "generate_llm_prediction_tool":
                                try:
                                    payload = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                                    pred = payload.get("prediction_result")
                                except Exception:
                                    pass
                            if hasattr(msg, 'name') and msg.name == "calculate_confidence_metrics_tool":
                                try:
                                    payload = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                                    conf = payload.get("confidence_metrics")
                                except Exception:
                                    pass
                            if hasattr(msg, 'name') and msg.name == "generate_recommendation_tool":
                                try:
                                    payload = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                                    rec = payload.get("recommendation")
                                except Exception:
                                    pass

                        if pred is not None:
                            result["prediction_result"] = pred
                            result["status"] = "success"
                            result["next_agent"] = get_next_agent(agent_name)
                        if conf is not None:
                            result["confidence_metrics"] = conf
                        if rec is not None:
                            result["recommendation"] = rec

                    state.update(result)
                    return state
            
            # If no messages, return the original state with error
            print(f"❌ {agent_name} failed: No messages returned")
            state["status"] = "error"
            state["error"] = f"{agent_name} failed: No messages returned"
            return state
            
        except Exception as e:
            print(f"❌ {agent_name} failed: {str(e)}")
            state["status"] = "error"
            state["error"] = f"{agent_name} failed: {str(e)}"
            return state
    
    return wrapper

def get_next_agent(current_agent: str) -> str:
    """Get the next agent in the workflow."""
    workflow_sequence = [
        "orchestrator",
        "data_collector", 
        "technical_analyzer",
        "sentiment_analyzer",
        "sentiment_integrator",
        "prediction_agent",
        "evaluator_optimizer",
        "elicitation"
    ]
    
    try:
        current_index = workflow_sequence.index(current_agent)
        if current_index + 1 < len(workflow_sequence):
            return workflow_sequence[current_index + 1]
        else:
            return "elicitation"  # Final step
    except ValueError:
        return "error_handler"  # Unknown agent

@lru_cache(maxsize=1)
def get_available_llm_client():
    """
    Get the best available LLM client with automatic fallback and quota handling.
    
    Returns:
        tuple: (llm_client, client_name, quota_status)
    """
    # Allow disabling LLM usage entirely (useful for tests/offline runs)
    disable_flag = os.getenv("DISABLE_LLM", "").strip().lower()
    if disable_flag in ("1", "true", "yes", "on"):
        quota_status = {
            "gemini": {"available": False, "reason": "Disabled by DISABLE_LLM"},
            "groq": {"available": False, "reason": "Disabled by DISABLE_LLM"},
        }
        try:
            print("LLM usage disabled via DISABLE_LLM env var; running in offline mode")
        except Exception:
            pass
        return None, None, quota_status
    quota_status = {
        "gemini": {"available": False, "reason": "Not tested"},
        "groq": {"available": False, "reason": "Not tested"}
    }

    selected_client = None
    selected_name = None

    # Probe Groq
    try:
        groq_client = get_groq_client()
        try:
            print("Groq client available")
        except Exception:
            pass
        quota_status["groq"]["available"] = True
        quota_status["groq"]["reason"] = "Available"
        # Prefer Groq by default
        selected_client = groq_client
        selected_name = "groq"
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            try:
                print("Groq quota limit reached")
            except Exception:
                pass
            quota_status["groq"]["available"] = False
            quota_status["groq"]["reason"] = "Quota limit reached"
        else:
            try:
                print(f"Groq client not available: {e}")
            except Exception:
                pass
            quota_status["groq"]["available"] = False
            quota_status["groq"]["reason"] = f"Error: {e}"

    # Probe Gemini as well so the UI can show accurate status,
    # but avoid triggering long provider-side retries by not making
    # test calls; client construction is fast-fail.
    try:
        gemini_client = get_gemini_client()
        try:
            print("Gemini client available")
        except Exception:
            pass
        quota_status["gemini"]["available"] = True
        quota_status["gemini"]["reason"] = "Available"
        if selected_client is None:
            selected_client = gemini_client
            selected_name = "gemini"
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg or "ResourceExhausted" in error_msg:
            try:
                print("Gemini quota limit reached")
            except Exception:
                pass
            quota_status["gemini"]["available"] = False
            quota_status["gemini"]["reason"] = "Quota limit reached"
        else:
            try:
                print(f"Gemini client not available: {e}")
            except Exception:
                pass
            quota_status["gemini"]["available"] = False
            quota_status["gemini"]["reason"] = f"Error: {e}"

    if selected_client is None:
        try:
            print("No LLM clients available")
        except Exception:
            pass
        return None, None, quota_status

    return selected_client, selected_name, quota_status

def display_quota_status(quota_status: dict):
    """
    Display quota status information to the user.
    
    Args:
        quota_status: Dictionary containing quota status for each provider
    """
    try:
        print("\nLLM Provider Status:")
    except Exception:
        pass
    print("=" * 40)
    
    for provider, status in quota_status.items():
        line = f"{provider.upper()}: {status['reason']}"
        try:
            print(line)
        except Exception:
            pass
    
    try:
        print("=" * 40)
    except Exception:
        pass

@lru_cache(maxsize=1)
def build_graph():
    """
    Build the LangGraph workflow for stock prediction following official LangGraph protocol.
    
    🚀 ENTRY POINT: orchestrator
    📊 WORKFLOW FLOW:z
    1. 🎯 Orchestrator - Initializes and coordinates the process (with tools)
    2. 📈 Data Collector - Fetches stock data, company info, market data (with tools)
    3. 🔍 Technical Analyzer - Performs technical analysis (with tools)
    4. 📰 Sentiment Analyzer - Analyzes news and social media sentiment
    5. 🔗 Sentiment Integrator - Combines technical and sentiment analysis
    6. 🤖 Prediction Agent - Makes final predictions using LLMs
    7. 📊 Evaluator Optimizer - Evaluates prediction quality
    8. ✅ Elicitation - Final confirmation and summary
    🏁 EXIT POINT: END (after elicitation or on error)
    """
    
    # Create the state graph with schema
    workflow = StateGraph(AgentState)
    
    # Get LLM clients with automatic fallback and quota handling
    llm_client, client_name, quota_status = get_available_llm_client()
    
    if not llm_client:
        print("❌ No LLM clients available - using fallback workflow")
        # Create a simple fallback workflow without LLM agents
        workflow.add_node("orchestrator", orchestrator.orchestrate)
        workflow.add_node("data_collector", data_collector.collect_data)
        workflow.add_node("technical_analyzer", technical_analyzer.analyze_technical)
        workflow.add_node("sentiment_analyzer", sentiment_analyzer.analyze_sentiment)
        workflow.add_node("sentiment_integrator", sentiment_integrator.integrate_sentiment)
        workflow.add_node("prediction_agent", prediction_agent.make_prediction)
        workflow.add_node("evaluator_optimizer", evaluator_optimizer.evaluate)
        workflow.add_node("elicitation", elicitation.elicit_confirmation)
        
        workflow.set_entry_point("orchestrator")
        
        # Simple linear flow
        workflow.add_conditional_edges("orchestrator", lambda x: "data_collector")
        workflow.add_conditional_edges("data_collector", lambda x: "technical_analyzer")
        workflow.add_conditional_edges("technical_analyzer", lambda x: "sentiment_analyzer")
        workflow.add_conditional_edges("sentiment_analyzer", lambda x: "sentiment_integrator")
        workflow.add_conditional_edges("sentiment_integrator", lambda x: "prediction_agent")
        workflow.add_conditional_edges("prediction_agent", lambda x: "evaluator_optimizer")
        workflow.add_conditional_edges("evaluator_optimizer", lambda x: "elicitation")
        workflow.add_conditional_edges("elicitation", lambda x: END)
        
        return workflow.compile()
    
    try:
        print(f"Using {client_name.upper()} LLM client")
    except Exception:
        pass
    
    # Create proper LangGraph agents with tools using create_react_agent
    orchestrator_agent = create_react_agent(
        llm_client,
        [
            validate_ticker_symbol,
            check_market_status,
            determine_analysis_parameters,
            initialize_workflow_state,
            coordinate_workflow_stage,
            handle_workflow_error
        ],
        prompt="""You are an orchestrator agent for stock prediction. Your job is to validate tickers, check market status, and initialize the workflow.

IMPORTANT: You MUST use the available tools to perform these tasks. Follow this exact sequence:

1. First, call validate_ticker_symbol with the ticker from the user's request
2. Then, call check_market_status to check current market conditions
3. Finally, call initialize_workflow_state with the ticker and timeframe

Do not proceed without using these tools. Always use the tools to get the required information.

User request: {input}"""
    )
    
    data_collector_agent = create_react_agent(
        llm_client,
        [
            collect_price_data,
            collect_company_info,
            collect_market_data,
            calculate_technical_indicators,
            validate_data_quality,
            collect_comprehensive_data
        ],
        prompt="""You are a data collector agent for stock prediction. Your job is to collect comprehensive data including price data, company info, market data, and technical indicators.

IMPORTANT: You MUST use the available tools to perform these tasks. Start by calling collect_comprehensive_data to get all the data you need.

Do not proceed without using these tools. Always use the tools to collect the required data.

User request: {input}"""
    )
    
    technical_analyzer_agent = create_react_agent(
        llm_client,
        [
            calculate_advanced_indicators,
            identify_chart_patterns,
            analyze_support_resistance,
            perform_trend_analysis,
            generate_trading_signals,
            validate_technical_analysis,
            calculate_short_term_indicators,
            compute_supertrend,
            compute_ichimoku_cloud,
            compute_keltner_channels,
            compute_donchian_channels,
        ],
        prompt="""You are a technical analyzer agent for stock prediction. Your job is to perform comprehensive technical analysis including indicators, patterns, support/resistance, trend analysis, and trading signals.

IMPORTANT: You MUST use the available tools to perform these tasks. Follow this sequence:

1. Call calculate_advanced_indicators to get core indicators
2. Then call perform_trend_analysis to analyze trends
3. Call calculate_short_term_indicators for 1-day focused metrics
4. Optionally compute_supertrend, compute_ichimoku_cloud, compute_keltner_channels, and compute_donchian_channels for additional confirmation
5. Finally call generate_trading_signals to generate trading signals

Do not proceed without using these tools. Always use the tools to perform the analysis.

User request: {input}"""
    )

    prediction_agent_tools = [
        generate_llm_prediction_tool,
        generate_rule_based_prediction_tool,
        calculate_confidence_metrics_tool,
        generate_recommendation_tool,
    ]
    prediction_agent_react = create_react_agent(
        llm_client,
        prediction_agent_tools,
        prompt="""You are the prediction agent. You MUST:

1) Call generate_llm_prediction_tool with the comprehensive analysis summary you receive in messages.
2) If that fails, call generate_rule_based_prediction_tool as fallback.
3) Then call calculate_confidence_metrics_tool with technical_analysis, sentiment_integration, and the prediction_result.
4) Finally, call generate_recommendation_tool with the prediction_result and confidence_metrics.

Return a concise final message summarizing direction, confidence, and recommended action.

User request: {input}"""
    )
    
    # Add nodes. Keep stable direct-call nodes for earlier stages; use wrapper for prediction.
    workflow.add_node("orchestrator", orchestrator.orchestrate_with_tools)  # 🎯 Orchestrator with direct tool calls
    workflow.add_node("data_collector", data_collector.collect_data)  # 📈 Data Collector direct
    workflow.add_node("technical_analyzer", create_llm_agent_wrapper(technical_analyzer_agent, "technical_analyzer"))  # 🔍 Technical Analyzer with tools
    # Always use tool-enabled prediction agent wrapper which calls tools deterministically
    workflow.add_node("prediction_agent", create_llm_agent_wrapper(prediction_agent_react, "prediction_agent"))  # 🤖 Prediction Agent with tools
    
    # Add traditional nodes (without tools)
    workflow.add_node("sentiment_analyzer", sentiment_analyzer.analyze_sentiment)  # 📰 Sentiment Analyzer
    workflow.add_node("sentiment_integrator", sentiment_integrator.integrate_sentiment)  # 🔗 Sentiment Integrator
    workflow.add_node("evaluator_optimizer", evaluator_optimizer.evaluate)  # 📊 Evaluator Optimizer
    workflow.add_node("elicitation", elicitation.elicit_confirmation)  # ✅ Elicitation
    
    # 🚀 SET ENTRY POINT
    workflow.set_entry_point("orchestrator")
    
    # Define the flow using conditional edges for proper LangGraph protocol
    
    # 🎯 Orchestrator → 📈 Data Collector (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "orchestrator",
        lambda x: "data_collector" if x.get("status") == "success" else END
    )
    
    # 📈 Data Collector → 🔍 Technical Analyzer (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "data_collector",
        lambda x: "technical_analyzer" if x.get("status") == "success" else END
    )
    
    # 🔍 Technical Analyzer → 📰 Sentiment Analyzer (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "technical_analyzer",
        lambda x: "sentiment_analyzer" if x.get("status") == "success" else END
    )
    
    # 📰 Sentiment Analyzer → 🔗 Sentiment Integrator (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "sentiment_analyzer",
        lambda x: "sentiment_integrator" if x.get("status") == "success" else END
    )
    
    # 🔗 Sentiment Integrator → 🤖 Prediction Agent (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "sentiment_integrator",
        lambda x: "prediction_agent" if x.get("status") == "success" else END
    )
    
    # 🤖 Prediction Agent → 📊 Evaluator Optimizer (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "prediction_agent",
        lambda x: "evaluator_optimizer" if x.get("status") == "success" else END
    )
    
    # 📊 Evaluator Optimizer → ✅ Elicitation (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "evaluator_optimizer",
        lambda x: "elicitation" if x.get("status") == "success" else END
    )
    
    # ✅ Elicitation → 🏁 EXIT (final step)
    workflow.add_conditional_edges(
        "elicitation",
        lambda x: END
    )
    
    return workflow.compile()

@lru_cache(maxsize=1)
def build_chatbot_graph():
    """
    Build a LangGraph workflow that starts with the chatbot and follows official LangGraph protocol.
    
    🚀 ENTRY POINT: chatbot
    📊 WORKFLOW FLOW:
    1. 💬 Chatbot - Processes user input and determines action
    2. 🎯 Orchestrator - Initializes and coordinates the process (with tools)
    3. 📈 Data Collector - Fetches stock data, company info, market data (with tools)
    4. 🔍 Technical Analyzer - Performs technical analysis (with tools)
    5. 📰 Sentiment Analyzer - Analyzes news and social media sentiment
    6. 🔗 Sentiment Integrator - Combines technical and sentiment analysis
    7. 🤖 Prediction Agent - Makes final predictions using LLMs
    8. 📊 Evaluator Optimizer - Evaluates prediction quality
    9. ✅ Elicitation - Final confirmation and summary
    🏁 EXIT POINT: END (after elicitation or on error)
    """
    
    # Create the state graph with schema
    workflow = StateGraph(AgentState)
    
    # Get LLM clients with automatic fallback and quota handling
    llm_client, client_name, quota_status = get_available_llm_client()
    
    if not llm_client:
        print("❌ No LLM clients available - using fallback workflow")
        # Create a simple fallback workflow without LLM agents
        workflow.add_node("chatbot", chatbot_agent.chatbot_node)
        workflow.add_node("orchestrator", orchestrator.orchestrate)
        workflow.add_node("data_collector", data_collector.collect_data)
        workflow.add_node("technical_analyzer", technical_analyzer.analyze_technical)
        workflow.add_node("sentiment_analyzer", sentiment_analyzer.analyze_sentiment)
        workflow.add_node("sentiment_integrator", sentiment_integrator.integrate_sentiment)
        workflow.add_node("prediction_agent", prediction_agent.make_prediction)
        workflow.add_node("evaluator_optimizer", evaluator_optimizer.evaluate)
        workflow.add_node("elicitation", elicitation.elicit_confirmation)
        
        workflow.set_entry_point("chatbot")
        
        # Simple linear flow
        workflow.add_conditional_edges("chatbot", lambda x: "orchestrator" if x.get("next_agent") == "workflow_orchestrator" else END)
        workflow.add_conditional_edges("orchestrator", lambda x: "data_collector")
        workflow.add_conditional_edges("data_collector", lambda x: "technical_analyzer")
        workflow.add_conditional_edges("technical_analyzer", lambda x: "sentiment_analyzer")
        workflow.add_conditional_edges("sentiment_analyzer", lambda x: "sentiment_integrator")
        workflow.add_conditional_edges("sentiment_integrator", lambda x: "prediction_agent")
        workflow.add_conditional_edges("prediction_agent", lambda x: "evaluator_optimizer")
        workflow.add_conditional_edges("evaluator_optimizer", lambda x: "elicitation")
        workflow.add_conditional_edges("elicitation", lambda x: END)
        
        return workflow.compile()
    
    print(f"🤖 Using {client_name.upper()} LLM client")
    
    # Create proper LangGraph agents with tools using create_react_agent
    orchestrator_agent = create_react_agent(
        llm_client,
        [
            validate_ticker_symbol,
            check_market_status,
            determine_analysis_parameters,
            initialize_workflow_state,
            coordinate_workflow_stage,
            handle_workflow_error
        ],
        prompt="You are an orchestrator agent for stock prediction. Your job is to validate tickers, check market status, and initialize the workflow. You MUST use the available tools to perform these tasks. Start by calling validate_ticker_symbol, then check_market_status, and finally initialize_workflow_state. Do not proceed without using these tools."
    )
    
    data_collector_agent = create_react_agent(
        llm_client,
        [
            collect_price_data,
            collect_company_info,
            collect_market_data,
            calculate_technical_indicators,
            validate_data_quality,
            collect_comprehensive_data
        ],
        prompt="You are a data collector agent for stock prediction. Your job is to collect comprehensive data including price data, company info, market data, and technical indicators. You MUST use the available tools to perform these tasks. Start by calling collect_comprehensive_data to get all the data you need. Do not proceed without using these tools."
    )
    
    technical_analyzer_agent = create_react_agent(
        llm_client,
        [
            calculate_advanced_indicators,
            identify_chart_patterns,
            analyze_support_resistance,
            perform_trend_analysis,
            generate_trading_signals,
            validate_technical_analysis
        ],
        prompt="You are a technical analyzer agent for stock prediction. Your job is to perform comprehensive technical analysis including indicators, patterns, support/resistance, trend analysis, and trading signals. You MUST use the available tools to perform these tasks. Start by calling calculate_advanced_indicators, then perform_trend_analysis, and finally generate_trading_signals. Do not proceed without using these tools."
    )
    
    # Add nodes
    workflow.add_node("chatbot", chatbot_agent.chatbot_node)  # 💬 Chatbot
    # Use deterministic, non‑LLM nodes for early stages
    workflow.add_node("orchestrator", orchestrator.orchestrate_with_tools)  # 🎯 Orchestrator without LLM
    workflow.add_node("data_collector", data_collector.collect_data)  # 📈 Data Collector without LLM
    workflow.add_node("technical_analyzer", create_llm_agent_wrapper(technical_analyzer_agent, "technical_analyzer"))  # 🔍 Technical Analyzer deterministic wrapper (no LLM)

    # Traditional nodes (no LLM)
    workflow.add_node("sentiment_analyzer", sentiment_analyzer.analyze_sentiment)
    workflow.add_node("sentiment_integrator", sentiment_integrator.integrate_sentiment)
    # Only prediction agent uses LLM
    workflow.add_node("prediction_agent", prediction_agent.make_prediction)
    workflow.add_node("evaluator_optimizer", evaluator_optimizer.evaluate)
    workflow.add_node("elicitation", elicitation.elicit_confirmation)
    
    # 🚀 SET ENTRY POINT
    workflow.set_entry_point("chatbot")
    
    # Define the flow using conditional edges for proper LangGraph protocol
    
    # 💬 Chatbot → 🎯 Orchestrator (if stock analysis needed) or 🏁 EXIT
    workflow.add_conditional_edges(
        "chatbot",
        lambda x: "orchestrator" if x.get("next_agent") == "workflow_orchestrator" else END
    )
    
    # 🎯 Orchestrator → 📈 Data Collector (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "orchestrator",
        lambda x: "data_collector" if x.get("status") == "success" else END
    )
    
    # 📈 Data Collector → 🔍 Technical Analyzer (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "data_collector",
        lambda x: "technical_analyzer" if x.get("status") == "success" else END
    )
    
    # 🔍 Technical Analyzer → 📰 Sentiment Analyzer (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "technical_analyzer",
        lambda x: "sentiment_analyzer" if x.get("status") == "success" else END
    )
    
    # 📰 Sentiment Analyzer → 🔗 Sentiment Integrator (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "sentiment_analyzer",
        lambda x: "sentiment_integrator" if x.get("status") == "success" else END
    )
    
    # 🔗 Sentiment Integrator → 🤖 Prediction Agent (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "sentiment_integrator",
        lambda x: "prediction_agent" if x.get("status") == "success" else END
    )
    
    # 🤖 Prediction Agent → 📊 Evaluator Optimizer (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "prediction_agent",
        lambda x: "evaluator_optimizer" if x.get("status") == "success" else END
    )
    
    # 📊 Evaluator Optimizer → ✅ Elicitation (or 🏁 EXIT on error)
    workflow.add_conditional_edges(
        "evaluator_optimizer",
        lambda x: "elicitation" if x.get("status") == "success" else END
    )
    
    # ✅ Elicitation → 🏁 EXIT (final step)
    workflow.add_conditional_edges(
        "elicitation",
        lambda x: END
    )
    
    return workflow.compile()

def run_prediction(ticker: str, timeframe: str = "1d", *args, **kwargs):
    """
    Backward/forward compatible run_prediction:
    - Positional args: (ticker, timeframe, [low_api_mode], [fast_ta_mode])
    - Keyword args: low_api_mode: bool, fast_ta_mode: bool
    """
    # Extract flags from args/kwargs with safe defaults
    low_api_mode = bool(kwargs.get("low_api_mode", False))
    fast_ta_mode = bool(kwargs.get("fast_ta_mode", False))
    use_ml_model = bool(kwargs.get("use_ml_model", False))
    if len(args) >= 1:
        low_api_mode = bool(args[0])
    if len(args) >= 2:
        fast_ta_mode = bool(args[1])
    if len(args) >= 3:
        use_ml_model = bool(args[2])
    """
    🚀 ENTRY POINT: Run the complete stock prediction pipeline following official LangGraph protocol.
    
    Args:
        ticker: Stock ticker symbol
        timeframe: Prediction timeframe
        
    Returns:
        Complete prediction results from the workflow
    """
    graph = build_graph()
    
    # Get quota status for user information
    _, _, quota_status = get_available_llm_client()
    display_quota_status(quota_status)
    
    # Initialize the workflow with input data and messages for LangGraph agents
    input_data = {
        "ticker": ticker,
        "timeframe": timeframe,
        "timestamp": None,
        "status": "initialized",
        "messages": [
            {
                "role": "user",
                "content": f"Analyze stock {ticker} with timeframe {timeframe}. Validate the ticker, check market status, and initialize the workflow."
            }
        ],
        "input": f"Analyze stock {ticker} with timeframe {timeframe}. Validate the ticker, check market status, and initialize the workflow."
    }
    
    # Check result cache (15-minute TTL)
    cache_key = f"prediction_result::{ticker}::{timeframe}::lowapi={int(low_api_mode)}::fastta={int(fast_ta_mode)}::ml={int(use_ml_model)}"
    cached = get_cached_result(cache_key, ttl_seconds=15 * 60)
    if cached:
        cached["quota_status"] = quota_status
        return cached

    # 🚀 INVOKE THE WORKFLOW
    input_data["low_api_mode"] = low_api_mode
    input_data["fast_ta_mode"] = fast_ta_mode
    input_data["use_ml_model"] = use_ml_model
    result = graph.invoke(input_data)
    
    # Add quota status to result for user information
    result["quota_status"] = quota_status

    # Safety net: if prediction_result is missing (e.g., early exit), run prediction locally
    try:
        if not result.get("prediction_result") and result.get("status") == "success":
            from agents.prediction_agent import PredictionAgent
            pa = PredictionAgent()
            pred_res = pa.make_prediction(result)
            # Merge keys back into result
            if isinstance(pred_res, dict):
                result.update(pred_res)
    except Exception:
        pass

    # Store in cache
    try:
        set_cached_result(cache_key, result)
    except Exception:
        pass
    
    return result

def run_chatbot_workflow(user_query: str):
    """
    🚀 ENTRY POINT: Run the chatbot-powered workflow following official LangGraph protocol.
    
    Args:
        user_query: User's natural language query
        
    Returns:
        Complete results including chatbot response and workflow results
    """
    graph = build_chatbot_graph()
    
    # Get quota status for user information
    _, _, quota_status = get_available_llm_client()
    display_quota_status(quota_status)
    
    # Initialize the workflow with user query and messages for LangGraph agents
    input_data = {
        "user_query": user_query,
        "timestamp": None,
        "status": "initialized",
        "messages": [
            {
                "role": "user",
                "content": user_query
            }
        ]
    }
    
    # Try cache if the query is a stock analysis asking for a specific ticker/timeframe
    # Very simple heuristic: extract the first ALLCAPS token up to 5 chars as ticker and assume timeframe 1d
    tokens = user_query.split()
    guessed_ticker = next((t for t in tokens if t.isupper() and len(t) <= 5), None)
    cache_key = None
    if guessed_ticker:
        cache_key = f"prediction_result::{guessed_ticker}::1d"
        cached = get_cached_result(cache_key, ttl_seconds=15 * 60)
        if cached:
            cached["quota_status"] = quota_status
            return cached

    # 🚀 INVOKE THE CHATBOT WORKFLOW
    result = graph.invoke(input_data)
    
    # Add quota status to result for user information
    result["quota_status"] = quota_status
    # Cache if we recognized a ticker
    if cache_key:
        try:
            set_cached_result(cache_key, result)
        except Exception:
            pass
    
    return result

def visualize_graph(save_path: str = "workflow_graph.png"):
    """
    🎨 Visualize the LangGraph workflow as a directed graph.
    
    Args:
        save_path: Path to save the visualization image
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Define nodes with their labels and colors (without emojis for font compatibility)
    nodes = {
        "ENTRY": {"color": "lightgreen", "shape": "s"},
        "Orchestrator": {"color": "lightblue", "shape": "o"},
        "Data Collector": {"color": "lightcoral", "shape": "o"},
        "Technical Analyzer": {"color": "lightyellow", "shape": "o"},
        "Sentiment Analyzer": {"color": "lightpink", "shape": "o"},
        "Sentiment Integrator": {"color": "lightcyan", "shape": "o"},
        "Prediction Agent": {"color": "lightgray", "shape": "o"},
        "Evaluator Optimizer": {"color": "lightsteelblue", "shape": "o"},
        "Elicitation": {"color": "lightgreen", "shape": "o"},
        "EXIT": {"color": "red", "shape": "s"}
    }
    
    # Add nodes
    for node, attrs in nodes.items():
        G.add_node(node, **attrs)
    
    # Define edges (success paths)
    success_edges = [
        ("Orchestrator", "Data Collector"),
        ("Data Collector", "Technical Analyzer"),
        ("Technical Analyzer", "Sentiment Analyzer"),
        ("Sentiment Analyzer", "Sentiment Integrator"),
        ("Sentiment Integrator", "Prediction Agent"),
        ("Prediction Agent", "Evaluator Optimizer"),
        ("Evaluator Optimizer", "Elicitation"),
        ("Elicitation", "EXIT")
    ]
    
    # Add success edges
    for edge in success_edges:
        G.add_edge(edge[0], edge[1], color="green", style="solid")
    
    # Add error edges (from each node to EXIT)
    error_nodes = [
        "Orchestrator", "Data Collector", "Technical Analyzer",
        "Sentiment Analyzer", "Sentiment Integrator", 
        "Prediction Agent", "Evaluator Optimizer"
    ]
    
    for node in error_nodes:
        G.add_edge(node, "EXIT", color="red", style="dashed")
    
    # Create the visualization
    plt.figure(figsize=(16, 12))
    
    # Position nodes using hierarchical layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes
    for node in G.nodes():
        node_attrs = G.nodes[node]
        color = node_attrs.get("color", "lightblue")
        shape = node_attrs.get("shape", "o")
        
        if shape == "s":  # Square for entry/exit
            nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                 node_color=color, node_size=3000, 
                                 node_shape='s')
        else:  # Circle for agents
            nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                 node_color=color, node_size=2500)
    
    # Draw edges
    edge_colors = [G[u][v]["color"] for u, v in G.edges()]
    edge_styles = [G[u][v]["style"] for u, v in G.edges()]
    
    # Draw solid edges (success paths)
    solid_edges = [(u, v) for u, v in G.edges() if G[u][v]["style"] == "solid"]
    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, 
                          edge_color="green", width=2, 
                          arrows=True, arrowsize=20)
    
    # Draw dashed edges (error paths)
    dashed_edges = [(u, v) for u, v in G.edges() if G[u][v]["style"] == "dashed"]
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, 
                          edge_color="red", width=1, 
                          style="dashed", arrows=True, arrowsize=15)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='lightgreen', label='Entry/Exit Points'),
        mpatches.Patch(color='lightblue', label='Agent Nodes'),
        mpatches.Patch(color='green', label='Success Paths'),
        mpatches.Patch(color='red', label='Error Paths (Dashed)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    # Add title
    plt.title("Agentic Stock Prediction Workflow", fontsize=16, fontweight="bold")
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎨 Workflow visualization saved to: {save_path}")
    return save_path

def print_graph_info():
    """
    📊 Print detailed information about the workflow graph.
    """
    print("🤖 Agentic Stock Prediction Workflow")
    print("=" * 50)
    
    print("\n🚀 ENTRY POINTS:")
    print("  • run_prediction(ticker, timeframe)")
    print("  • orchestrator node")
    
    print("\n📊 AGENT NODES:")
    agents = [
        ("🎯 Orchestrator", "Initializes and coordinates the process"),
        ("📈 Data Collector", "Fetches stock data, company info, market data"),
        ("🔍 Technical Analyzer", "Performs technical analysis"),
        ("📰 Sentiment Analyzer", "Analyzes news and social media sentiment"),
        ("🔗 Sentiment Integrator", "Combines technical and sentiment analysis"),
        ("🤖 Prediction Agent", "Makes final predictions using LLMs"),
        ("📊 Evaluator Optimizer", "Evaluates prediction quality"),
        ("✅ Elicitation", "Final confirmation and summary")
    ]
    
    for agent, description in agents:
        print(f"  • {agent}: {description}")
    
    print("\n🏁 EXIT POINTS:")
    print("  • END (LangGraph constant)")
    print("  • Error exits from any agent")
    print("  • Normal completion after elicitation")
    
    print("\n🔄 FLOW PATHS:")
    print("  • Success Path: orchestrator → data_collector → technical_analyzer →")
    print("    sentiment_analyzer → sentiment_integrator → prediction_agent →")
    print("    evaluator_optimizer → elicitation → END")
    print("  • Error Paths: Any agent → END (on error)")
