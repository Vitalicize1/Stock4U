# 📚 Documentation Index

Welcome to the **Agentic Stock Predictor v2** documentation! This folder contains comprehensive guides for all aspects of the system.

## 🎯 **Quick Start**

1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Set API Keys:** Configure your environment variables
3. **Run Dashboard:** `streamlit run dashboard.py`
4. **Test Chatbot:** Navigate to the "💬 Chatbot" tab

## 📋 **Documentation Guide**

### 🤖 **Chatbot & AI Agents**
- **[CHATBOT_GUIDE.md](CHATBOT_GUIDE.md)** - Complete chatbot interface guide
- **[CHATBOT_NODE_GUIDE.md](CHATBOT_NODE_GUIDE.md)** - LangGraph node architecture
- **[TAVILY_INTEGRATION_GUIDE.md](TAVILY_INTEGRATION_GUIDE.md)** - Web search integration

### 🎨 **Visualization & Workflow**
- **[VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md)** - Workflow visualization guide
- **[WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md)** - Text-based workflow diagram

## 🚀 **System Architecture**

### **Core Components:**
- **LangGraph Workflow** - 8-agent prediction pipeline
- **Chatbot Interface** - Natural language interaction
- **Tavily Search** - Real-time web information
- **Streamlit Dashboard** - Web-based interface

### **Agent Pipeline:**
1. 🎯 **Orchestrator** - Initializes and coordinates
2. 📈 **Data Collector** - Fetches stock data
3. 🔍 **Technical Analyzer** - Technical analysis
4. 📰 **Sentiment Analyzer** - Sentiment analysis
5. 🔗 **Sentiment Integrator** - Combines analyses
6. 🤖 **Prediction Agent** - LLM predictions
7. 📊 **Evaluator Optimizer** - Evaluates results
8. ✅ **Elicitation** - Final confirmation

## 💡 **Key Features**

### **🤖 Chatbot Capabilities:**
- ✅ **Stock Analysis** - "Analyze AAPL stock"
- ✅ **Web Search** - "Search for latest Tesla news"
- ✅ **Workflow Info** - "How does the workflow work?"
- ✅ **Help & Guidance** - "What can you do?"

### **🔍 Web Search Integration:**
- ✅ **Real-time Information** - Current news and data
- ✅ **Multiple Search Types** - Stock, company, news, technical
- ✅ **Formatted Results** - Professional output with sources
- ✅ **Error Handling** - Graceful failures

### **🎨 Visualization:**
- ✅ **Workflow Diagrams** - Visual representation
- ✅ **Interactive Dashboard** - Streamlit interface
- ✅ **Multiple Formats** - PNG, HTML, text diagrams

## 🛠️ **Setup & Configuration**

### **Required API Keys:**
```bash
# OpenAI API (for LLM predictions)
OPENAI_API_KEY=sk-your-openai-key

# Google AI (alternative LLM)
GOOGLE_API_KEY=your-google-key

# Tavily Search (for web search)
TAVILY_API_KEY=tvly-your-tavily-key
```

### **Installation:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the system
streamlit run dashboard.py
```

## 🧪 **Testing**

### **Test Scripts:**
```bash
# Test chatbot functionality
python test_chatbot.py

# Test Tavily search integration
python test_tavily.py

# Test workflow visualization
python visualize_workflow.py
```

## 📊 **Usage Examples**

### **Chatbot Commands:**
```
"Analyze AAPL stock"           # Stock analysis
"Search for latest Tesla news" # Web search
"How does the workflow work?"  # System info
"What can you do?"            # Help
```

### **Dashboard Features:**
- 📊 **Predictions Tab** - Stock analysis interface
- 🎨 **Workflow Tab** - Visual workflow diagrams
- 💬 **Chatbot Tab** - Interactive AI assistant
- 📈 **Market Data Tab** - Real-time market data

## 🔧 **Development**

### **File Structure:**
```
agentic_stock_predictorv2/
├── agents/                    # AI agent modules
├── docs/                     # Documentation (this folder)
├── langgraph_flow.py         # Main workflow
├── dashboard.py              # Streamlit interface
├── main.py                   # Command-line interface
└── requirements.txt          # Dependencies
```

### **Adding New Features:**
1. **Create Agent Module** - Add to `agents/` folder
2. **Update Workflow** - Modify `langgraph_flow.py`
3. **Update Documentation** - Add guide to `docs/`
4. **Test Integration** - Run test scripts

## 🎯 **Next Steps**

1. **Configure API Keys** - Set up your environment
2. **Run the System** - Start with `streamlit run dashboard.py`
3. **Explore Features** - Try different chatbot commands
4. **Customize** - Modify agents and workflows as needed

## 📞 **Support**

For questions or issues:
1. Check the relevant documentation guide
2. Run test scripts to verify functionality
3. Review error messages and logs
4. Check API key configuration

---

**Happy predicting! 🚀📈** 