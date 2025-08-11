# 🎨 LangGraph Workflow Visualization Guide

## 📊 Available Visualization Methods

### 1. **Command Line Visualization**
```bash
python visualize_workflow.py
```
- ✅ Generates `workflow_graph.png`
- ✅ Prints detailed workflow information
- ✅ Shows entry/exit points and agent descriptions

### 2. **Streamlit Dashboard Integration**
```bash
streamlit run dashboard.py
```
- ✅ Interactive web interface
- ✅ "🎨 Workflow" tab with visualization
- ✅ Click "🔄 Generate Workflow Diagram" button
- ✅ Displays workflow information and diagram

### 3. **HTML Viewer**
```bash
# Open in browser
workflow_viewer.html
```
- ✅ Beautiful HTML interface
- ✅ Responsive design
- ✅ Detailed workflow explanation
- ✅ Color-coded information sections

## 🎯 Visualization Features

### 📈 **Graph Elements:**
- **Square Nodes:** Entry/Exit points (green/red)
- **Circle Nodes:** Agent nodes (different colors)
- **Green Arrows:** Success paths between agents
- **Red Dashed Arrows:** Error paths to exit
- **Arrows:** Show flow direction

### 🎨 **Color Coding:**
- **Light Green:** Entry/Exit points
- **Light Blue:** Orchestrator
- **Light Coral:** Data Collector
- **Light Yellow:** Technical Analyzer
- **Light Pink:** Sentiment Analyzer
- **Light Cyan:** Sentiment Integrator
- **Light Gray:** Prediction Agent
- **Light Steel Blue:** Evaluator Optimizer
- **Red:** Exit point

### 🔄 **Flow Paths:**
```
Success Path: ENTRY → Orchestrator → Data Collector → Technical Analyzer 
    → Sentiment Analyzer → Sentiment Integrator → Prediction Agent 
    → Evaluator Optimizer → Elicitation → EXIT

Error Paths: Any Agent → EXIT (on error)
```

## 🛠️ Technical Implementation

### **Dependencies:**
```bash
pip install networkx matplotlib
```

### **Functions Available:**
```python
from langgraph_flow import visualize_graph, print_graph_info

# Generate visualization
image_path = visualize_graph("workflow_graph.png")

# Print workflow information
print_graph_info()
```

### **Integration Points:**
- ✅ **Main workflow:** `langgraph_flow.py`
- ✅ **Dashboard:** `dashboard.py` (Streamlit)
- ✅ **Standalone script:** `visualize_workflow.py`
- ✅ **HTML viewer:** `workflow_viewer.html`

## 📋 Usage Examples

### **Quick Visualization:**
```python
from langgraph_flow import visualize_graph
visualize_graph("my_workflow.png")
```

### **Dashboard Integration:**
```python
# In Streamlit app
if st.button("Generate Workflow"):
    image_path = visualize_graph("workflow_dashboard.png")
    st.image(image_path, use_column_width=True)
```

### **Command Line:**
```bash
# Generate visualization
python visualize_workflow.py

# Run dashboard
streamlit run dashboard.py

# Open HTML viewer
start workflow_viewer.html
```

## 🎯 Key Benefits

1. **Visual Understanding:** See the complete workflow at a glance
2. **Error Handling:** Visualize error paths and exit points
3. **Agent Relationships:** Understand how agents connect
4. **Flow Control:** See conditional routing logic
5. **Documentation:** Self-documenting workflow structure

## 🔧 Customization Options

### **Modify Node Colors:**
```python
nodes = {
    "Orchestrator": {"color": "lightblue", "shape": "o"},
    # Add more nodes with custom colors
}
```

### **Add Custom Edges:**
```python
success_edges = [
    ("Orchestrator", "Data Collector"),
    # Add more edges
]
```

### **Change Layout:**
```python
# Use different layout algorithms
pos = nx.spring_layout(G, k=3, iterations=50)
# or
pos = nx.kamada_kawai_layout(G)
```

## 📁 File Structure

```
Stock4U/
├── langgraph_flow.py          # Main workflow + visualization
├── visualize_workflow.py      # Standalone visualization script
├── dashboard.py               # Streamlit dashboard with visualization
├── workflow_viewer.html       # HTML viewer
├── workflow_graph.png         # Generated visualization
├── WORKFLOW_DIAGRAM.md       # Text-based diagram
└── VISUALIZATION_GUIDE.md    # This guide
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install networkx matplotlib
   ```

2. **Generate visualization:**
   ```bash
   python visualize_workflow.py
   ```

3. **View in dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

4. **Open HTML viewer:**
   ```bash
   start workflow_viewer.html
   ```

## ✅ Success Indicators

- ✅ **PNG file generated:** `workflow_graph.png`
- ✅ **No font warnings:** Clean visualization
- ✅ **All nodes visible:** 8 agents + entry/exit
- ✅ **Color coding:** Different colors for different agents
- ✅ **Flow arrows:** Clear direction indicators
- ✅ **Error paths:** Dashed red lines to exit

Your LangGraph workflow is now **fully visualized** with multiple viewing options! 🎨 