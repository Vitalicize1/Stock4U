# 🔍 Tavily Search Integration Guide

## 🎯 **Overview**

Your LangGraph workflow now includes **Tavily Search integration** for real-time web search capabilities. This allows your chatbot to search the web for current information about stocks, companies, market news, and more.

## 🚀 **Setup Instructions**

### **1. Get Tavily API Key**
1. Visit [Tavily AI](https://tavily.com/)
2. Sign up for a free account
3. Get your API key from the dashboard

### **2. Set Environment Variable**
```bash
# Set your Tavily API key
export TAVILY_API_KEY="tvly-your-api-key-here"
```

Or add to your `.env` file:
```
TAVILY_API_KEY=tvly-your-api-key-here
```

### **3. Install Dependencies**
```bash
pip install langchain-tavily
```

## 🏗️ **Architecture**

### **Tavily Search Agent (`agents/tavily_search_agent.py`)**
```python
class TavilySearchAgent:
    def search_stock_info(self, ticker: str) -> Dict[str, Any]
    def search_company_info(self, company_name: str) -> Dict[str, Any]
    def search_market_news(self, topic: str) -> Dict[str, Any]
    def search_technical_analysis(self, ticker: str) -> Dict[str, Any]
    def search_sentiment_analysis(self, ticker: str) -> Dict[str, Any]
    def search_custom_query(self, query: str) -> Dict[str, Any]
```

### **Chatbot Integration**
The chatbot now automatically detects web search requests and uses Tavily to provide real-time information.

## 📊 **Search Types**

### **1. Stock Information Search**
```python
# Search for stock-specific information
search_agent.search_stock_info("AAPL")
# Query: "AAPL stock price news analysis financial performance"
```

### **2. Company Information Search**
```python
# Search for company information
search_agent.search_company_info("Apple Inc")
# Query: "Apple Inc company financial performance earnings revenue"
```

### **3. Market News Search**
```python
# Search for market news and trends
search_agent.search_market_news("AI stocks")
# Query: "AI stocks latest news trends analysis"
```

### **4. Technical Analysis Search**
```python
# Search for technical analysis information
search_agent.search_technical_analysis("TSLA")
# Query: "TSLA technical analysis chart patterns indicators"
```

### **5. Sentiment Analysis Search**
```python
# Search for sentiment analysis
search_agent.search_sentiment_analysis("NVDA")
# Query: "NVDA sentiment analysis social media investor sentiment"
```

### **6. Custom Query Search**
```python
# Search for any custom query
search_agent.search_custom_query("latest cryptocurrency news")
```

## 💬 **Chatbot Usage**

### **Web Search Commands:**
- `"Search for latest Apple news"`
- `"Find information about Tesla"`
- `"What is the latest AI news?"`
- `"Tell me about NVIDIA company"`
- `"Search for cryptocurrency trends"`
- `"Find latest market analysis"`

### **Example Interactions:**
```
User: "Search for latest Apple news"
Bot: 🔍 Web Search Results for: "latest Apple news"
     📊 Found 3 results (Response time: 1.23s)
     
     1. Apple Reports Strong Q4 Earnings
     Apple Inc. reported better-than-expected fourth-quarter earnings...
     📎 Source: https://example.com/apple-earnings
     
     2. New iPhone Launch Expected
     Apple is preparing to launch its latest iPhone model...
     📎 Source: https://example.com/iphone-launch
```

## 🔧 **Technical Implementation**

### **1. Search Agent Initialization:**
```python
from agents.tavily_search_agent import TavilySearchAgent

# Initialize with custom max results
search_agent = TavilySearchAgent(max_results=5)
```

### **2. LangGraph Node Integration:**
```python
def tavily_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # Extract search parameters from state
    search_type = state.get("search_type", "stock_info")
    ticker = state.get("ticker", "")
    
    # Perform search based on type
    if search_type == "stock_info" and ticker:
        search_results = search_agent.search_stock_info(ticker)
    
    # Update state with results
    state.update({
        "tavily_search_results": search_results,
        "status": "success"
    })
    
    return state
```

### **3. Chatbot Integration:**
```python
def chatbot_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # Process user query
    response = chatbot.process_user_query(user_input, state)
    
    # Handle web search requests
    if response.get("response_type") == "web_search":
        search_results = get_web_search_results(user_input)
        formatted_response = chatbot.format_web_search_response(user_input, search_results)
        response["message"] = formatted_response
```

## 🧪 **Testing**

### **Run Integration Tests:**
```bash
python test_tavily.py
```

### **Test Output:**
```
🧪 Testing Tavily Search Integration
==================================================

1️⃣ Testing TavilySearchAgent initialization...
✅ TavilySearchAgent initialized successfully

2️⃣ Testing stock info search...
✅ Stock search completed: 3 results
   Status: success

3️⃣ Testing custom query search...
✅ Custom search completed: 3 results
   Status: success

4️⃣ Testing integration function...
✅ Integration function completed: 3 results
   Status: success

5️⃣ Sample search results:
   Title: Apple Stock Analysis
   URL: https://example.com/apple-stock
   Content preview: Apple Inc. (AAPL) stock analysis shows strong performance...

✅ All Tavily search tests completed successfully!
```

## 📋 **Response Format**

### **Successful Search Response:**
```python
{
    "status": "success",
    "query": "AAPL stock news",
    "results": [
        {
            "title": "Apple Stock Analysis",
            "url": "https://example.com/apple-stock",
            "content": "Apple Inc. (AAPL) stock analysis shows...",
            "score": 0.85
        }
    ],
    "response_time": 1.23,
    "total_results": 3,
    "summary": "1. **Apple Stock Analysis**\n   Apple Inc. (AAPL) stock analysis...\n   Source: https://example.com/apple-stock"
}
```

### **Error Response:**
```python
{
    "status": "error",
    "query": "invalid query",
    "error": "API rate limit exceeded",
    "results": [],
    "total_results": 0
}
```

## 🎯 **Key Features**

### **1. Real-time Information:**
- ✅ **Live web search** for current information
- ✅ **Multiple search types** (stock, company, news, technical, sentiment)
- ✅ **Custom queries** for any topic

### **2. Smart Integration:**
- ✅ **Automatic detection** of search requests in chatbot
- ✅ **Formatted responses** with titles, content, and sources
- ✅ **Error handling** with graceful failures

### **3. Performance:**
- ✅ **Fast response times** (typically 1-3 seconds)
- ✅ **Configurable results** (1-10 results per search)
- ✅ **Response time tracking** for monitoring

### **4. LangGraph Compliance:**
- ✅ **Native LangGraph node** integration
- ✅ **State management** with search results
- ✅ **Conditional routing** based on search success

## 🚀 **Usage Examples**

### **1. Stock Analysis with Web Data:**
```python
# Search for stock information
result = search_agent.search_stock_info("TSLA")
print(f"Found {result['total_results']} results for Tesla stock")
```

### **2. Market News Search:**
```python
# Search for market news
result = search_agent.search_market_news("AI technology stocks")
print(f"Found {result['total_results']} AI stock news articles")
```

### **3. Chatbot Web Search:**
```python
# Use in chatbot
from langgraph_flow import run_chatbot_workflow

result = run_chatbot_workflow("Search for latest Apple news")
# Returns formatted web search results
```

## 🔍 **Advanced Configuration**

### **1. Custom Search Parameters:**
```python
# Initialize with custom settings
search_agent = TavilySearchAgent(max_results=10)

# Custom search with specific parameters
results = search_agent.search_custom_query("latest tech news 2024")
```

### **2. Error Handling:**
```python
try:
    results = search_agent.search_stock_info("AAPL")
    if results["status"] == "success":
        print(f"Found {results['total_results']} results")
    else:
        print(f"Search failed: {results['error']}")
except Exception as e:
    print(f"Search error: {str(e)}")
```

## ✅ **Success Indicators**

- ✅ **API key configured** and working
- ✅ **Search results returned** with titles and content
- ✅ **Response times** under 5 seconds
- ✅ **Chatbot integration** working correctly
- ✅ **Error handling** graceful and informative
- ✅ **Formatted responses** readable and professional

## 🎯 **Benefits**

1. **Real-time Information** - Access current news and data
2. **Enhanced Stock Analysis** - Combine technical analysis with web data
3. **Market Intelligence** - Stay updated with latest trends
4. **Comprehensive Responses** - Provide rich, sourced information
5. **Professional Output** - Formatted results with proper attribution

## 🔗 **Integration Points**

- ✅ **Chatbot Agent** - Automatic web search detection
- ✅ **LangGraph Workflow** - Native node integration
- ✅ **Dashboard** - Web search results display
- ✅ **State Management** - Unified state with search results
- ✅ **Error Handling** - Consistent error management

Your **Tavily search integration is now fully functional** and ready to provide real-time web information to your stock prediction system! 🔍🚀 