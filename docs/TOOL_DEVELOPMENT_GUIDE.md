# 🛠️ LangGraph Tool Development Guide

## 📋 **Overview**

This guide ensures all future tools follow the proper LangGraph structure with ToolNode integration and conditional routing.

## 🏗️ **Tool Structure Requirements**

### **1. Tool Decorator Pattern**
```python
from langchain_core.tools import tool

@tool
def your_tool_name(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Clear description of what this tool does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 (default: 10)
        
    Returns:
        Dictionary with tool results
    """
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

### **2. Tool Organization**
```python
# agents/your_agent_tools.py
"""
Your Agent Tools for Stock Prediction Workflow

This module provides tools for the your_agent to:
- Function 1
- Function 2
- Function 3
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.tools import tool

class YourAgentTools:
    """Tools for the your_agent to perform specific functions."""
    
    def __init__(self):
        """Initialize your agent tools."""
        pass

@tool
def tool_function_1(param: str) -> Dict[str, Any]:
    """Description of tool function 1."""
    # Implementation
    pass

@tool
def tool_function_2(param: str) -> Dict[str, Any]:
    """Description of tool function 2."""
    # Implementation
    pass
```

### **3. Agent Integration**
```python
# agents/your_agent.py
def your_agent_with_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your agent that uses tools for comprehensive analysis.
    
    Args:
        state: Current state containing data and parameters
        
    Returns:
        Updated state with analysis results
    """
    try:
        # Check if we need to execute tools
        needs_tools = state.get("needs_tools", True)
        
        if needs_tools:
            print("🔧 Your agent needs tools - routing to your_agent_tools")
            return {
                "status": "success",
                "needs_tools": True,
                "current_tool_node": "your_agent_tools",
                "next_agent": "your_agent_tools"
            }
        
        # Tool execution logic
        result1 = tool_function_1.invoke(param1)
        result2 = tool_function_2.invoke(param2)
        
        # Update state
        state.update({
            "status": "success",
            "needs_tools": False,
            "next_agent": "next_agent",
            "your_agent_tools_used": [
                "tool_function_1",
                "tool_function_2"
            ]
        })
        
        return state
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Your agent failed: {str(e)}",
            "next_agent": "error_handler",
            "needs_tools": False
        }
```

### **4. LangGraph Integration**
```python
# langgraph_flow.py
from agents.tools.your_agent_tools import (
    tool_function_1,
    tool_function_2
)

def build_graph():
    # Create ToolNode for your agent's tools
    your_agent_tools = ToolNode([
        tool_function_1,
        tool_function_2
    ], name="your_agent_tools")
    
    # Add nodes
    workflow.add_node("your_agent", your_agent.your_agent_with_tools)
    workflow.add_node("your_agent_tools", your_agent_tools)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "your_agent",
        lambda x: "your_agent_tools" if x.get("needs_tools") else "next_agent"
    )
    
    workflow.add_conditional_edges(
        "your_agent_tools",
        lambda x: "next_agent"
    )
```

## 📊 **State Schema Updates**

### **Add to AgentState TypedDict:**
```python
class AgentState(TypedDict):
    # ... existing fields ...
    
    # Your agent specific fields
    your_agent_data: Optional[Dict[str, Any]]
    your_agent_tools_used: Optional[list]
    
    # Tool execution tracking (already present)
    tool_calls: Optional[list]
    tool_results: Optional[Dict[str, Any]]
    needs_tools: Optional[bool]
    current_tool_node: Optional[str]
```

## 🔄 **Tool Execution Flow**

### **Standard Flow:**
```
Agent → Check needs_tools → ToolNode → Agent (continue)
```

### **Error Flow:**
```
Agent → Error → Error Handler → END
```

## ✅ **Tool Development Checklist**

- [ ] **Tool Decorator**: Use `@tool` decorator
- [ ] **Type Hints**: Proper type annotations
- [ ] **Docstring**: Clear description and parameters
- [ ] **Error Handling**: Try/catch with proper error response
- [ ] **Return Format**: Consistent dictionary structure
- [ ] **Agent Integration**: `needs_tools` flag and routing
- [ ] **ToolNode Registration**: Add to appropriate ToolNode
- [ ] **State Updates**: Update AgentState schema if needed
- [ ] **Testing**: Unit tests for tool functionality
- [ ] **Documentation**: Update relevant documentation

## 🎯 **Best Practices**

### **1. Tool Naming**
```python
# ✅ Good
@tool
def collect_stock_data(ticker: str) -> Dict[str, Any]:

# ❌ Bad
@tool
def get_data(symbol: str) -> Dict[str, Any]:
```

### **2. Error Handling**
```python
# ✅ Good
try:
    result = perform_operation()
    return {"status": "success", "data": result}
except Exception as e:
    return {"status": "error", "error": str(e)}
```

### **3. State Management**
```python
# ✅ Good
state.update({
    "status": "success",
    "needs_tools": False,
    "next_agent": "next_agent",
    "tools_used": ["tool1", "tool2"]
})
```

### **4. Tool Organization**
```python
# ✅ Good - Organized by agent
agents/
├── data_collector_tools.py
├── technical_analyzer_tools.py
├── sentiment_analyzer_tools.py
└── your_agent_tools.py
```

## 🚀 **Example: New Sentiment Tool**

```python
# agents/sentiment_analyzer_tools.py
@tool
def analyze_news_sentiment(ticker: str, days: int = 7) -> Dict[str, Any]:
    """
    Analyze news sentiment for a stock over the specified number of days.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days to analyze (default: 7)
        
    Returns:
        Dictionary with sentiment analysis results
    """
    try:
        # Implementation
        sentiment_data = fetch_news_sentiment(ticker, days)
        
        return {
            "status": "success",
            "data": sentiment_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
```

## 📈 **Testing Tools**

### **Unit Test Template:**
```python
# tests/test_your_tools.py
import pytest
from agents.tools.your_agent_tools import tool_function_1

def test_tool_function_1():
    """Test tool_function_1 with valid input."""
    result = tool_function_1.invoke("AAPL")
    assert result["status"] == "success"
    assert "data" in result

def test_tool_function_1_error():
    """Test tool_function_1 with invalid input."""
    result = tool_function_1.invoke("")
    assert result["status"] == "error"
    assert "error" in result
```

## 🔧 **Integration Testing**

### **Workflow Test:**
```python
def test_your_agent_integration():
    """Test your agent with ToolNode integration."""
    from langgraph_flow import build_graph
    
    graph = build_graph()
    result = graph.invoke({
        "ticker": "AAPL",
        "timeframe": "1d",
        "needs_tools": True
    })
    
    assert result["status"] == "success"
    assert "your_agent_data" in result
```

## 📚 **Documentation Requirements**

1. **Tool Description**: What the tool does
2. **Parameters**: Input parameters and types
3. **Returns**: Output structure and types
4. **Examples**: Usage examples
5. **Error Cases**: Common error scenarios
6. **Integration**: How it fits in the workflow

## 🎯 **Summary**

Following this guide ensures:
- ✅ **Consistent Tool Structure**: All tools follow the same pattern
- ✅ **Proper LangGraph Integration**: ToolNode and conditional routing
- ✅ **Error Handling**: Robust error management
- ✅ **State Management**: Proper state updates and tracking
- ✅ **Testability**: Tools are easily testable
- ✅ **Maintainability**: Clear organization and documentation

This structure provides a solid foundation for scalable tool development within the LangGraph framework.
