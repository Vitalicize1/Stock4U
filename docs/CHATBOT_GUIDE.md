# 💬 Chatbot Interface Guide

## 🤖 AI Stock Prediction Chatbot

Your LangGraph workflow now has a **fully functional chatbot interface** that can interact with users and provide stock analysis through the 8-agent workflow.

## 🚀 How to Run the Chatbot

### 1. **Integrated Dashboard Chatbot**
```bash
streamlit run dashboard.py
```
- Navigate to the "💬 Chatbot" tab
- Interactive chat interface
- Integrated with the main dashboard

### 2. **Standalone Chatbot Interface**
```bash
streamlit run chatbot_interface.py
```
- Dedicated chatbot interface
- Full-screen chat experience
- Quick action buttons in sidebar

### 3. **Command Line Testing**
```bash
python test_chatbot.py
```
- Test chatbot functionality
- Verify responses without UI
- Debug and development tool

## 🎯 Chatbot Capabilities

### 📊 **Stock Analysis**
- **Analyze any stock ticker** (e.g., "Analyze AAPL")
- **Get predictions and technical analysis**
- **View sentiment analysis and risk assessment**
- **Real-time LangGraph workflow execution**

### 🤖 **Workflow Information**
- **Explain how the LangGraph workflow works**
- **Describe each AI agent's role**
- **Show the complete prediction pipeline**
- **Technical details and flow diagrams**

### 📈 **Market Data & Help**
- **Get real-time stock data**
- **View technical indicators**
- **Access company information**
- **General assistance and guidance**

## 💡 Example Conversations

### **Stock Analysis:**
```
User: "Analyze AAPL stock"
Bot: 📊 AAPL Analysis Results:
     🎯 Prediction: UP
     📊 Confidence: 63.9%
     💡 Recommendation: BUY
     📈 Technical Score: 90.0/100
     ⚠️ Risk Level: medium
```

### **Workflow Questions:**
```
User: "How does the workflow work?"
Bot: 🤖 LangGraph Workflow Overview:
     Our stock prediction system uses 8 specialized AI agents:
     1. 🎯 Orchestrator - Initializes and coordinates
     2. 📈 Data Collector - Fetches stock data
     3. 🔍 Technical Analyzer - Performs technical analysis
     ...
```

### **General Help:**
```
User: "What can you do?"
Bot: 🤖 I can help you with:
     📊 Stock Analysis
     🤖 Workflow Information
     📈 Market Data
     💡 Examples and guidance
```

## 🔧 Technical Implementation

### **Response Generation Logic:**
```python
def generate_chatbot_response(prompt: str) -> str:
    prompt_lower = prompt.lower()
    
    # Stock analysis requests
    if any(word in prompt_lower for word in ["analyze", "stock", "prediction"]):
        # Extract ticker and run LangGraph workflow
        result = run_prediction(ticker_match, "1d")
        return format_stock_analysis_response(ticker_match, result)
    
    # Workflow questions
    elif any(word in prompt_lower for word in ["workflow", "agents", "process"]):
        return workflow_explanation_response()
    
    # General help
    elif any(word in prompt_lower for word in ["help", "what can you do"]):
        return capabilities_response()
    
    # Default response
    else:
        return default_greeting_response()
```

### **Integration with LangGraph:**
- ✅ **Direct workflow execution** via `run_prediction()`
- ✅ **Real-time analysis** using all 8 agents
- ✅ **Error handling** and graceful failures
- ✅ **Formatted responses** with detailed results

## 🎨 User Interface Features

### **Streamlit Chat Interface:**
- ✅ **Chat message history** with session state
- ✅ **User and assistant message bubbles**
- ✅ **Real-time typing indicators**
- ✅ **Sidebar controls** for quick actions

### **Quick Action Buttons:**
- 🔄 **Clear Chat** - Reset conversation
- 📊 **Quick AAPL Analysis** - Instant stock analysis
- 🤖 **Workflow Info** - System explanation
- 💡 **Help** - Capabilities overview

### **Responsive Design:**
- ✅ **Mobile-friendly** interface
- ✅ **Wide layout** for better readability
- ✅ **Markdown formatting** for rich responses
- ✅ **Emoji and visual elements**

## 📋 Supported Commands

### **Stock Analysis Commands:**
- `"Analyze AAPL"`
- `"What's the prediction for TSLA?"`
- `"Stock analysis for GOOGL"`
- `"Price prediction MSFT"`

### **Workflow Information:**
- `"How does the workflow work?"`
- `"Explain the process"`
- `"What are the agents?"`
- `"Tell me about the system"`

### **General Help:**
- `"What can you do?"`
- `"Help"`
- `"Capabilities"`
- `"Examples"`

## 🔍 Advanced Features

### **Ticker Detection:**
- ✅ **Automatic ticker extraction** from natural language
- ✅ **Case-insensitive matching** (AAPL, aapl, Apple)
- ✅ **Error handling** for invalid tickers
- ✅ **Suggestions** for proper format

### **Response Formatting:**
- ✅ **Rich markdown** formatting
- ✅ **Emoji indicators** for different sections
- ✅ **Structured data** presentation
- ✅ **Professional styling**

### **Error Handling:**
- ✅ **Graceful failures** with helpful messages
- ✅ **Network error** handling
- ✅ **Invalid input** validation
- ✅ **User-friendly** error messages

## 🚀 Quick Start

### **1. Run the Dashboard:**
```bash
streamlit run dashboard.py
# Navigate to "💬 Chatbot" tab
```

### **2. Run Standalone Chatbot:**
```bash
streamlit run chatbot_interface.py
# Full chatbot interface
```

### **3. Test Functionality:**
```bash
python test_chatbot.py
# Verify responses work correctly
```

## 📁 File Structure

```
agentic_stock_predictorv2/
├── dashboard.py              # Main dashboard with chatbot tab
├── chatbot_interface.py      # Standalone chatbot interface
├── test_chatbot.py          # Chatbot testing script
├── langgraph_flow.py        # Core workflow + chatbot integration
└── CHATBOT_GUIDE.md        # This guide
```

## ✅ Success Indicators

- ✅ **Chatbot responds** to stock analysis requests
- ✅ **LangGraph workflow** executes successfully
- ✅ **Formatted responses** display correctly
- ✅ **Error handling** works gracefully
- ✅ **UI is responsive** and user-friendly
- ✅ **Session state** maintains chat history

## 🎯 Key Benefits

1. **Natural Language Interface** - Users can ask questions in plain English
2. **Real-time Analysis** - Direct integration with LangGraph workflow
3. **Educational Tool** - Explains the AI system and workflow
4. **User-Friendly** - Intuitive chat interface with helpful responses
5. **Extensible** - Easy to add new capabilities and responses

Your **LangGraph workflow now has a fully functional chatbot** that can interact with users and provide real-time stock analysis! 💬🤖 