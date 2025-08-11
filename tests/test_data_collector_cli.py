#!/usr/bin/env python3
"""
Test specifically the data collector agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_collector():
    """Test the data collector agent specifically."""
    
    print("🔍 Testing Data Collector Agent")
    print("=" * 40)
    
    try:
        from langgraph.prebuilt import create_react_agent
        from llm.gemini_client import get_gemini_client
        from agents.tools.data_collector_tools import collect_comprehensive_data
        
        # Get LLM client
        llm_client = get_gemini_client()
        
        # Create data collector agent
        data_collector_agent = create_react_agent(
            llm_client,
            [collect_comprehensive_data],
            prompt="You are a data collector agent for stock prediction. You MUST use the collect_comprehensive_data tool to get data for AAPL. Do not proceed without using this tool."
        )
        
        # Create initial state
        initial_state = {
            "ticker": "AAPL",
            "timeframe": "1d",
            "messages": [
                {
                    "role": "user",
                    "content": "Collect comprehensive data for AAPL stock"
                }
            ]
        }
        
        print("\n1. Running data collector agent...")
        result = data_collector_agent.invoke(initial_state)
        
        print(f"\n2. Result keys: {list(result.keys())}")
        
        if 'messages' in result:
            print(f"\n3. Messages:")
            for i, msg in enumerate(result['messages']):
                print(f"   Message {i}: {type(msg).__name__}")
                if hasattr(msg, 'name'):
                    print(f"   Name: {msg.name}")
                if hasattr(msg, 'content'):
                    print(f"   Content: {msg.content[:200]}...")
                if hasattr(msg, 'tool_calls'):
                    print(f"   Tool calls: {len(msg.tool_calls) if msg.tool_calls else 0}")
        
        return result
        
    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_data_collector()
