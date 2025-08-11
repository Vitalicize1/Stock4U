# tests/test_tool_integration.py
"""
Test file for LangGraph tool integration following official protocol.

This file tests the proper integration of tools with LangGraph using create_react_agent.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_flow import build_graph, build_chatbot_graph
from langgraph.prebuilt import create_react_agent
from llm.gemini_client import get_gemini_client
from agents.tools.data_collector_tools import collect_price_data, collect_company_info

def test_langgraph_protocol_structure():
    """Test that the LangGraph protocol structure is properly set up."""
    print("🧪 Testing LangGraph protocol structure...")
    
    # Test 1: Check if graphs can be built (will fail without API keys, but that's expected)
    try:
        main_graph = build_graph()
        print("✅ Main graph built successfully")
    except ValueError as e:
        if "GOOGLE_API_KEY not found" in str(e):
            print("⚠️ Main graph test skipped - API key not configured")
        else:
            print(f"❌ Failed to build main graph: {e}")
            return False
    except ImportError as e:
        if "langchain_groq" in str(e):
            print("⚠️ Main graph test skipped - langchain_groq not installed")
        else:
            print(f"❌ Failed to build main graph: {e}")
            return False
    except Exception as e:
        print(f"❌ Failed to build main graph: {e}")
        return False
    
    try:
        chatbot_graph = build_chatbot_graph()
        print("✅ Chatbot graph built successfully")
    except ValueError as e:
        if "GOOGLE_API_KEY not found" in str(e):
            print("⚠️ Chatbot graph test skipped - API key not configured")
        else:
            print(f"❌ Failed to build chatbot graph: {e}")
            return False
    except ImportError as e:
        if "langchain_groq" in str(e):
            print("⚠️ Chatbot graph test skipped - langchain_groq not installed")
        else:
            print(f"❌ Failed to build chatbot graph: {e}")
            return False
    except Exception as e:
        print(f"❌ Failed to build chatbot graph: {e}")
        return False
    
    # Test 2: Check if create_react_agent can be created (will fail without API keys, but that's expected)
    try:
        gemini_client = get_gemini_client()
        test_agent = create_react_agent(
            gemini_client,
            [collect_price_data, collect_company_info],
            prompt="You are a test agent."
        )
        print("✅ create_react_agent created successfully")
    except ValueError as e:
        if "GOOGLE_API_KEY not found" in str(e):
            print("⚠️ create_react_agent test skipped - API key not configured")
        else:
            print(f"❌ Failed to create react agent: {e}")
            return False
    except ImportError as e:
        if "langchain_groq" in str(e):
            print("⚠️ create_react_agent test skipped - langchain_groq not installed")
        else:
            print(f"❌ Failed to create react agent: {e}")
            return False
    except Exception as e:
        print(f"❌ Failed to create react agent: {e}")
        return False
    
    print("🎉 All LangGraph protocol structure tests passed!")
    return True

def test_agent_state_schema():
    """Test that the AgentState schema includes required fields for LLM agents."""
    print("\n🧪 Testing AgentState schema...")
    
    try:
        from langgraph_flow import AgentState
        
        # Check if required fields exist
        required_fields = [
            "messages"  # Required for LLM agent tool calling
        ]
        
        for field in required_fields:
            if field in AgentState.__annotations__:
                print(f"✅ Field '{field}' present in AgentState")
            else:
                print(f"❌ Field '{field}' missing from AgentState")
                return False
        
        print("✅ All required LLM agent fields present in AgentState")
        return True
        
    except Exception as e:
        print(f"❌ AgentState schema test failed: {e}")
        return False

def test_tool_definition():
    """Test that tools are properly defined with @tool decorator."""
    print("\n🧪 Testing tool definitions...")
    
    try:
        # Test that tools have proper structure
        tools_to_test = [
            collect_price_data,
            collect_company_info
        ]
        
        for i, tool in enumerate(tools_to_test):
            print(f"Debug: Testing tool {i}: {tool}")
            print(f"Debug: type(tool) = {type(tool)}")
            
            # Check if tool is a LangChain StructuredTool
            if hasattr(tool, 'name'):
                print(f"✅ Tool '{tool.name}' is a proper LangChain tool")
            else:
                print(f"❌ Tool missing 'name' attribute")
                return False
            
            # Check if tool has description
            if hasattr(tool, 'description') and tool.description:
                print(f"✅ Tool '{tool.name}' has description")
            else:
                print(f"⚠️ Tool '{tool.name}' missing description")
            
            # Check if tool is callable
            if callable(tool):
                print(f"✅ Tool '{tool.name}' is callable")
            else:
                print(f"❌ Tool '{tool.name}' is not callable")
                return False
            
            # Check if tool has the proper LangChain tool structure
            if hasattr(tool, 'args_schema'):
                print(f"✅ Tool '{tool.name}' has proper schema")
            else:
                print(f"⚠️ Tool '{tool.name}' may not have proper schema")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool definition test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_client_integration():
    """Test that LLM clients can be properly integrated."""
    print("\n🧪 Testing LLM client integration...")
    
    try:
        from llm.gemini_client import get_gemini_client
        from llm.groq_client import get_groq_client
        
        # Test Gemini client (will fail without API key, but that's expected)
        try:
            gemini_client = get_gemini_client()
            print("✅ Gemini client created successfully")
        except ValueError as e:
            if "GOOGLE_API_KEY not found" in str(e):
                print("⚠️ Gemini client test skipped - API key not configured")
            else:
                print(f"❌ Gemini client creation failed: {e}")
                return False
        except ImportError as e:
            if "langchain_google_genai" in str(e):
                print("⚠️ Gemini client test skipped - langchain_google_genai not installed")
            else:
                print(f"❌ Gemini client import failed: {e}")
                return False
        
        # Test Groq client (will fail without API key, but that's expected)
        try:
            groq_client = get_groq_client()
            print("✅ Groq client created successfully")
        except ValueError as e:
            if "GROQ_API_KEY not found" in str(e):
                print("⚠️ Groq client test skipped - API key not configured")
            else:
                print(f"❌ Groq client creation failed: {e}")
                return False
        except ImportError as e:
            if "langchain_groq" in str(e):
                print("⚠️ Groq client test skipped - langchain_groq not installed")
            else:
                print(f"❌ Groq client import failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ LLM client integration test failed: {e}")
        return False

def run_all_tests():
    """Run all LangGraph protocol integration tests."""
    print("🚀 Running LangGraph Protocol Integration Tests\n")
    
    tests = [
        test_langgraph_protocol_structure,
        test_agent_state_schema,
        test_tool_definition,
        test_llm_client_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test {test.__name__} failed")
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LangGraph protocol integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    run_all_tests()
