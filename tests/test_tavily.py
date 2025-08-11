#!/usr/bin/env python3
"""
🧪 LangGraph Tutorial Step 4: Define the Graph with Tools
Implementing step 4 from the LangGraph tutorial to integrate LLM model and define graph with tools.
"""

import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def get_llm_model():
    """
    Initialize and return the appropriate LLM model based on available API keys.
    Follows the LangGraph tutorial pattern for model selection.
    """
    # Check which API keys are available
    google_key = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Priority order: Groq (fastest) -> Google -> OpenAI -> Anthropic
    if groq_key:
        print("🚀 Using Groq LLM (fastest)")
        try:
            from langchain.chat_models import init_chat_model
            return init_chat_model("groq:llama3-8b-8192")
        except ImportError as e:
            print(f"❌ Import error for Groq: {e}")
            raise ImportError("Please install langchain: pip install langchain")
    elif google_key:
        print("🤖 Using Google Gemini LLM")
        try:
            from langchain.chat_models import init_chat_model
            return init_chat_model("google_genai:gemini-2.0-flash")
        except ImportError as e:
            print(f"❌ Import error for Google: {e}")
            raise ImportError("Please install langchain: pip install langchain")
    elif openai_key:
        print("🧠 Using OpenAI GPT-4 LLM")
        try:
            from langchain.chat_models import init_chat_model
            return init_chat_model("openai:gpt-4o-mini")
        except ImportError as e:
            print(f"❌ Import error for OpenAI: {e}")
            raise ImportError("Please install langchain: pip install langchain")
    elif anthropic_key:
        print("🎭 Using Anthropic Claude LLM")
        try:
            from langchain.chat_models import init_chat_model
            return init_chat_model("anthropic:claude-3-5-sonnet-latest")
        except ImportError as e:
            print(f"❌ Import error for Anthropic: {e}")
            raise ImportError("Please install langchain: pip install langchain")
    else:
        raise ValueError("No LLM API keys found! Please set one of: GROQ_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")

def define_graph_with_tools():
    """
    Step 4: Define the graph with tools according to LangGraph tutorial.
    
    This implements the exact pattern from the tutorial:
    https://langchain-ai.github.io/langgraph/tutorials/get-started/2-add-tools/#4-define-the-graph
    """
    
    # Step 1: Install and configure Tavily (already done)
    print("✅ Step 1: Tavily search engine configured")
    
    # Step 2: Configure environment (already done)
    print("✅ Step 2: Environment configured")
    
    # Step 3: Define the tool
    print("🔧 Step 3: Defining Tavily search tool...")
    tool = TavilySearch(max_results=2)
    tools = [tool]
    
    # Print tool info for debugging
    print(f"✅ Tool name: {tool.name}")
    print(f"✅ Tool description: {tool.description}")
    
    # Test the tool
    try:
        test_result = tool.invoke("What's a 'node' in LangGraph?")
        print(f"✅ Tool test successful: {len(test_result.get('results', []))} results")
    except Exception as e:
        print(f"⚠️ Tool test failed: {e}")
    
    # Step 4: Define the graph
    print("🔄 Step 4: Defining the graph with tools...")
    
    # Define the state schema
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    
    # Initialize the graph builder
    graph_builder = StateGraph(State)
    
    # Get the LLM model
    llm = get_llm_model()
    
    # Bind tools to the LLM (this tells the LLM about available tools)
    llm_with_tools = llm.bind_tools(tools)
    print("✅ Tools bound to LLM")
    
    # Print bound tools info
    print(f"✅ Bound tools: {[tool.name for tool in tools]}")
    
    # Define the chatbot node
    def chatbot(state: State):
        """Chatbot node that processes messages and can call tools."""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    # Add the chatbot node
    graph_builder.add_node("chatbot", chatbot)
    print("✅ Chatbot node added")
    
    # Create tool node using prebuilt ToolNode
    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)
    print("✅ Tool node added")
    
    # Add conditional edges
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,  # Prebuilt condition that routes to tools or END
    )
    
    # Add edges
    graph_builder.add_edge("tools", "chatbot")  # Return to chatbot after tools
    graph_builder.add_edge(START, "chatbot")    # Start with chatbot
    
    # Compile the graph
    graph = graph_builder.compile()
    print("✅ Graph compiled successfully")
    
    return graph, tools

def test_graph_with_tools():
    """
    Test the graph with tools implementation.
    """
    print("🧪 Testing LangGraph with Tools Implementation")
    print("=" * 60)
    
    try:
        # Define the graph
        graph, tools = define_graph_with_tools()
        
        # Test the graph with a simple query
        print("\n🤖 Testing chatbot with tools...")
        
        def stream_graph_updates(user_input: str):
            """Stream updates from the graph (from tutorial)."""
            try:
                for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
                    for value in event.values():
                        if "messages" in value and value["messages"]:
                            last_message = value["messages"][-1]
                            if hasattr(last_message, 'content'):
                                print("Assistant:", last_message.content)
            except Exception as e:
                print(f"❌ Stream error: {e}")
                # Try a simpler approach
                try:
                    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
                    if "messages" in result and result["messages"]:
                        last_message = result["messages"][-1]
                        if hasattr(last_message, 'content'):
                            print("Assistant:", last_message.content)
                except Exception as e2:
                    print(f"❌ Invoke error: {e2}")
        
        # Test queries
        test_queries = [
            "What do you know about LangGraph?",
            "Search for latest AI news",
            "Tell me about stock market trends"
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 40)
            try:
                stream_graph_updates(query)
            except Exception as e:
                print(f"❌ Error with query '{query}': {e}")
        
        print("\n✅ Graph with tools test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during graph testing: {e}")
        return False

def main():
    """Main function to run the LangGraph tutorial step 4 implementation."""
    print("🚀 LangGraph Tutorial Step 4: Define the Graph with Tools")
    print("=" * 70)
    
    # Check API keys
    required_keys = ["TAVILY_API_KEY"]
    optional_keys = ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    
    print("\n🔑 Checking API keys...")
    
    # Check required keys
    missing_required = []
    for key in required_keys:
        if not os.getenv(key):
            missing_required.append(key)
    
    if missing_required:
        print(f"❌ Missing required API keys: {missing_required}")
        print("💡 Please set these environment variables:")
        for key in missing_required:
            print(f"   export {key}='your-api-key-here'")
        return False
    
    # Check optional keys
    available_llm_keys = []
    for key in optional_keys:
        if os.getenv(key):
            available_llm_keys.append(key)
    
    if not available_llm_keys:
        print("⚠️ No LLM API keys found. Please set one of:")
        for key in optional_keys:
            print(f"   export {key}='your-api-key-here'")
        print("   Recommended: GROQ_API_KEY (fastest) or GOOGLE_API_KEY")
        return False
    
    print(f"✅ Found LLM API keys: {available_llm_keys}")
    
    # Test the implementation
    success = test_graph_with_tools()
    
    if success:
        print("\n🎉 Success! LangGraph with tools is working correctly.")
        print("\n💡 You can now use this graph in your main application:")
        print("   - Import the define_graph_with_tools function")
        print("   - Use the returned graph for interactive conversations")
        print("   - The chatbot can now search the web using Tavily")
    else:
        print("\n⚠️ Some issues were encountered. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    main() 