# 🛠️ LangGraph Tool Integration Guide

## 📋 **Overview**

This guide explains how tools are properly integrated with LangGraph following the **official protocol** using `create_react_agent` and LLM-powered tool calling.

## 🏗️ **Architecture Overview**

### **Before (Manual Tool Routing):**
```
Agent → Check needs_tools → ToolNode → Agent (continue)
```

### **After (Official LangGraph Protocol):**
```
LLM Agent → Tool Calling → ToolNode → LLM Agent (continue)
```

## 🔧 **Official LangGraph Tool Integration Pattern**

### **1. Tool Definition**
Tools are defined using the `@tool` decorator:

```python
from langchain_core.tools import tool

@tool
def collect_price_data(ticker: str, period: str = "3mo") -> Dict[str, Any]:
    """
    Collect comprehensive historical price data for a stock.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary with comprehensive price data
    """
    try:
        # Tool implementation
        result = perform_data_collection(ticker, period)
        
        return {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

### **2. LLM Agent Creation with Tools**
Use `create_react_agent` to create LLM agents that can call tools:

```python
from langgraph.prebuilt import create_react_agent
from llm.gemini_client import get_gemini_client

# Get LLM client
gemini_client = get_gemini_client()

# Create agent with tools
data_collector_agent = create_react_agent(
    gemini_client,
    [
        collect_price_data,
        collect_company_info,
        collect_market_data,
        calculate_technical_indicators,
        validate_data_quality,
        collect_comprehensive_data
    ],
    prompt="You are a data collector agent for stock prediction. Your job is to collect comprehensive data including price data, company info, market data, and technical indicators. Use the available tools to perform these tasks."
)
```

### **3. LangGraph Flow Integration**
The workflow uses LLM agents that can call tools directly:

```python
def build_graph():
    """Build the LangGraph workflow with proper LLM agent tool integration."""
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Get LLM clients
    gemini_client = get_gemini_client()
    
    # Create LLM agents with tools
    orchestrator_agent = create_react_agent(
        gemini_client,
        [validate_ticker_symbol, check_market_status, ...],
        prompt="You are an orchestrator agent..."
    )
    
    data_collector_agent = create_react_agent(
        gemini_client,
        [collect_price_data, collect_company_info, ...],
        prompt="You are a data collector agent..."
    )
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("data_collector", data_collector_agent)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "orchestrator",
        lambda x: "data_collector" if x.get("status") == "success" else END
    )
    
    return workflow.compile()
```

## 📊 **State Schema**

The `AgentState` includes the `messages` field required for LLM agent tool calling:

```python
class AgentState(TypedDict):
    # ... existing fields ...
    
    # Messages for proper LangGraph tool integration
    messages: Optional[list]
```

## 🔄 **Tool Execution Flow**

### **Standard Flow:**
1. **LLM Agent Entry**: LLM agent receives message with task
2. **Tool Decision**: LLM decides which tools to call based on task
3. **Tool Execution**: LangGraph automatically executes tools via ToolNode
4. **Result Processing**: LLM processes tool results and continues

### **Error Flow:**
1. **LLM Agent Error**: LLM agent encounters error
2. **Error Handling**: Set status to error and route to END
3. **Error Recovery**: Error handler processes error and routes appropriately

## 🎯 **Benefits of Official LangGraph Protocol**

### **1. Proper LangGraph Integration**
- Uses `create_react_agent` for LLM-powered tool calling
- Follows official LangGraph patterns
- Automatic tool execution via ToolNode

### **2. LLM Intelligence**
- LLM decides which tools to call
- Natural language tool calling
- Context-aware tool selection

### **3. Scalability**
- Tools can be easily added/removed
- Multiple tools can be executed in parallel
- Tool execution is managed by LangGraph

### **4. Error Handling**
- Robust error handling at tool level
- Graceful degradation when tools fail
- Clear error propagation

## 🛠️ **Available LLM Agents with Tools**

### **1. Orchestrator Agent**
- `validate_ticker_symbol`: Validate stock ticker
- `check_market_status`: Check market open/close status
- `determine_analysis_parameters`: Set analysis parameters
- `initialize_workflow_state`: Initialize workflow state

### **2. Data Collector Agent**
- `collect_price_data`: Collect historical price data
- `collect_company_info`: Gather company information
- `collect_market_data`: Collect market data
- `calculate_technical_indicators`: Calculate technical indicators
- `validate_data_quality`: Validate data quality
- `collect_comprehensive_data`: Collect all data types

### **3. Technical Analyzer Agent**
- `calculate_advanced_indicators`: Calculate advanced technical indicators
- `identify_chart_patterns`: Identify chart patterns
- `analyze_support_resistance`: Analyze support/resistance levels
- `perform_trend_analysis`: Perform trend analysis
- `generate_trading_signals`: Generate trading signals
- `validate_technical_analysis`: Validate technical analysis

## 🔧 **Adding New Tools**

### **1. Create Tool Function**
```python
@tool
def your_new_tool(param1: str, param2: int = 10) -> Dict[str, Any]:
    """Description of your new tool."""
    try:
        # Tool implementation
        result = perform_operation(param1, param2)
        
        return {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

### **2. Add to LLM Agent**
```python
# In langgraph_flow.py
orchestrator_agent = create_react_agent(
    gemini_client,
    [
        existing_tool1,
        existing_tool2,
        your_new_tool  # Add your new tool here
    ],
    prompt="You are an orchestrator agent..."
)
```

## ✅ **Best Practices**

### **1. Tool Design**
- Use clear, descriptive names
- Include comprehensive docstrings
- Implement proper error handling
- Return consistent data structures

### **2. LLM Agent Integration**
- Provide clear prompts for agents
- Include all relevant tools in agent creation
- Use appropriate LLM clients

### **3. Error Handling**
- Catch exceptions in tools
- Return error status with details
- Propagate errors appropriately
- Provide fallback behavior

### **4. Testing**
- Test tools individually
- Test LLM agent-tool integration
- Test error scenarios
- Validate tool results

## 🚀 **Usage Examples**

### **Running the Workflow**
```python
from langgraph_flow import run_prediction

# Run prediction with proper LLM agent tool integration
result = run_prediction("AAPL", "1d")
print(f"Prediction: {result}")
```

### **Chatbot Integration**
```python
from langgraph_flow import run_chatbot_workflow

# Run chatbot with LLM agent tool integration
response = run_chatbot_workflow("Analyze AAPL stock")
print(f"Response: {response}")
```

This integration follows the **official LangGraph protocol** and ensures that all tools are properly managed by LLM agents that can intelligently decide when and how to use them.
