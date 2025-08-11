# 🚀 LangGraph Workflow - Entry & Exit Points

## 📊 Workflow Overview

```
                    🚀 ENTRY POINT
                         ↓
                    🎯 Orchestrator
                         ↓
                    📈 Data Collector
                         ↓
                    🔍 Technical Analyzer
                         ↓
                    📰 Sentiment Analyzer
                         ↓
                    🔗 Sentiment Integrator
                         ↓
                    🤖 Prediction Agent
                         ↓
                    📊 Evaluator Optimizer
                         ↓
                    ✅ Elicitation
                         ↓
                    🏁 EXIT POINT
```

## 🎯 Detailed Flow with Exit Points

```
🚀 ENTRY: run_prediction(ticker, timeframe)
    ↓
🎯 Orchestrator (Initializes workflow)
    ↓ (success) → 📈 Data Collector
    ↓ (error)   → 🏁 EXIT
    
📈 Data Collector (Fetches stock data)
    ↓ (success) → 🔍 Technical Analyzer
    ↓ (error)   → 🏁 EXIT
    
🔍 Technical Analyzer (Technical analysis)
    ↓ (success) → 📰 Sentiment Analyzer
    ↓ (error)   → 🏁 EXIT
    
📰 Sentiment Analyzer (Sentiment analysis)
    ↓ (success) → 🔗 Sentiment Integrator
    ↓ (error)   → 🏁 EXIT
    
🔗 Sentiment Integrator (Combines analyses)
    ↓ (success) → 🤖 Prediction Agent
    ↓ (error)   → 🏁 EXIT
    
🤖 Prediction Agent (LLM predictions)
    ↓ (success) → 📊 Evaluator Optimizer
    ↓ (error)   → 🏁 EXIT
    
📊 Evaluator Optimizer (Evaluates results)
    ↓ (success) → ✅ Elicitation
    ↓ (error)   → 🏁 EXIT
    
✅ Elicitation (Final confirmation)
    ↓ (always)  → 🏁 EXIT
```

## 🔧 Technical Implementation

### 🚀 Entry Points:
1. **`run_prediction(ticker, timeframe)`** - Main entry function
2. **`workflow.set_entry_point("orchestrator")`** - Graph entry node
3. **`graph.invoke(input_data)`** - Workflow invocation

### 🏁 Exit Points:
1. **`END`** - LangGraph's built-in exit constant
2. **Error exits** - When any agent returns `"status": "error"`
3. **Normal completion** - After elicitation completes

### 📋 State Flow:
```
Input Data → Orchestrator → Data Collector → Technical Analyzer 
    ↓              ↓              ↓                ↓
Sentiment Analyzer → Sentiment Integrator → Prediction Agent 
    ↓                    ↓                    ↓
Evaluator Optimizer → Elicitation → Final Results
```

## 🎯 Key Features:

- ✅ **Single Entry Point**: `orchestrator` node
- ✅ **Multiple Exit Points**: Error handling at each step
- ✅ **Conditional Routing**: Success/error paths
- ✅ **State Persistence**: Data flows through all agents
- ✅ **Error Recovery**: Graceful exit on failures

## 🔍 Usage Examples:

```python
# Entry point
result = run_prediction("AAPL", "1d")

# Exit points (automatic)
# - Success: Complete workflow with results
# - Error: Exit with error message
``` 