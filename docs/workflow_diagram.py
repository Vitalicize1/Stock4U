#!/usr/bin/env python3
"""
Workflow Diagram Generator for Stock Prediction Project
Generates structured visual representations of the project workflow using matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_main_workflow_diagram():
    """Create a comprehensive main workflow diagram."""
    
    # Set up the figure with better proportions
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define color scheme
    colors = {
        'input': '#E3F2FD',      # Light blue
        'data': '#F3E5F5',       # Light purple
        'analysis': '#E8F5E8',   # Light green
        'prediction': '#FFF3E0',  # Light orange
        'output': '#FCE4EC',     # Light pink
        'llm': '#E0F2F1',        # Light teal
        'sentiment': '#FFF8E1'   # Light yellow
    }
    
    # Define main workflow boxes
    workflow_boxes = [
        # Input Layer
        (5, 12.5, 2, 0.8, 'User Input\n(Ticker Symbol)', colors['input']),
        
        # Data Collection Layer
        (1, 10.5, 3, 1, 'Data Collector Agent\n• Price Data\n• Company Info\n• Market Data\n• Historical Data', colors['data']),
        
        # Analysis Layer - Technical
        (5, 10.5, 3, 1, 'Technical Analyzer Agent\n• 20+ Technical Indicators\n• Pattern Recognition\n• Support/Resistance\n• Trading Signals', colors['analysis']),
        (9, 10.5, 3, 1, 'Sentiment Analyzer Agent\n• News Sentiment\n• Social Media\n• Reddit Analysis\n• Market Sentiment', colors['sentiment']),
        
        # Integration Layer
        (5, 8.5, 3, 1, 'Sentiment Integration Agent\n• Signal Alignment\n• Confidence Adjustment\n• Risk Assessment\n• Integrated Scoring', colors['prediction']),
        
        # Prediction Layer
        (5, 6.5, 3, 1, 'Prediction Agent (LLM)\n• Gemini/Groq Integration\n• Direction Prediction\n• Confidence Scoring\n• Risk Assessment', colors['llm']),
        
        # Output Layer
        (1, 4.5, 2.5, 1, 'Chatbot Interface\n• Conversational AI\n• Natural Language\n• User-Friendly', colors['output']),
        (4.5, 4.5, 2.5, 1, 'Dashboard Interface\n• Visual Charts\n• Analysis Display\n• Interactive', colors['output']),
        (8, 4.5, 2.5, 1, 'API Endpoints\n• Programmatic Access\n• Integration Ready\n• Scalable', colors['output']),
        
        # Data Flow Labels
        (2.5, 9.5, 1, 0.4, 'Structured\nData Package', colors['data']),
        (6.5, 9.5, 1, 0.4, 'Technical\nAnalysis', colors['analysis']),
        (10.5, 9.5, 1, 0.4, 'Sentiment\nAnalysis', colors['sentiment']),
        (6.5, 7.5, 1, 0.4, 'Integrated\nAnalysis', colors['prediction']),
        (6.5, 5.5, 1, 0.4, 'Final\nPrediction', colors['llm']),
    ]
    
    # Draw workflow boxes
    for x, y, w, h, text, color in workflow_boxes:
        box = FancyBboxPatch((x, y), w, h, 
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=1.5)
        ax.add_patch(box)
        
        # Add text with better formatting
        ax.text(x + w/2, y + h/2, text, 
                ha='center', va='center', 
                fontsize=9, fontweight='bold',
                wrap=True)
    
    # Draw main workflow arrows
    main_arrows = [
        # Main flow
        ((6, 12.5), (6, 11.5)),      # Input to Data Collector
        ((2.5, 10.5), (2.5, 9.9)),   # Data Collector to Structured Data
        ((2.5, 9.5), (5, 9.5)),      # Structured Data to Technical Analyzer
        ((2.5, 9.5), (9, 9.5)),      # Structured Data to Sentiment Analyzer
        ((6.5, 10), (6.5, 9.5)),     # Technical Analyzer to Technical Analysis
        ((10.5, 10), (10.5, 9.5)),   # Sentiment Analyzer to Sentiment Analysis
        ((6.5, 9.5), (6.5, 8.9)),    # Technical Analysis to Sentiment Integration
        ((10.5, 9.5), (6.5, 8.9)),   # Sentiment Analysis to Sentiment Integration
        ((6.5, 8.5), (6.5, 7.9)),    # Sentiment Integration to Integrated Analysis
        ((6.5, 7.5), (6.5, 6.9)),    # Integrated Analysis to Prediction Agent
        ((6.5, 6.5), (6.5, 5.9)),    # Prediction Agent to Final Prediction
        ((6.5, 5.5), (2.25, 5.1)),   # Final Prediction to Chatbot
        ((6.5, 5.5), (5.75, 5.1)),   # Final Prediction to Dashboard
        ((6.5, 5.5), (9.25, 5.1)),   # Final Prediction to API
    ]
    
    for start, end in main_arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black", lw=2)
        ax.add_patch(arrow)
    
    # Add layer labels with better positioning
    layer_labels = [
        (6, 13.2, 'INPUT LAYER', 14, 'bold'),
        (6, 11.2, 'DATA COLLECTION LAYER', 14, 'bold'),
        (6, 9.2, 'ANALYSIS LAYER', 14, 'bold'),
        (6, 7.2, 'INTEGRATION LAYER', 14, 'bold'),
        (6, 5.2, 'PREDICTION LAYER', 14, 'bold'),
        (6, 3.2, 'OUTPUT LAYER', 14, 'bold'),
    ]
    
    for x, y, text, size, weight in layer_labels:
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=size, fontweight=weight, color='darkblue')
    
    # Add main title and subtitle
    ax.text(6, 13.8, 'STOCK PREDICTION PROJECT WORKFLOW', 
            ha='center', va='center', fontsize=18, fontweight='bold', color='darkred')
    
    ax.text(6, 13.5, 'Agentic Stock Predictor v2 - Multi-Agent AI System', 
            ha='center', va='center', fontsize=12, style='italic', color='darkblue')
    
    # Add key features box
    features_box = FancyBboxPatch((0.5, 1.5), 11, 2.5, 
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightyellow',
                                 edgecolor='orange',
                                 linewidth=2)
    ax.add_patch(features_box)
    
    features_text = """KEY FEATURES & CAPABILITIES:
• Multi-Agent Architecture with Specialized Agents
• LLM-Powered Predictions (Gemini/Groq Integration)
• Comprehensive Technical + Sentiment Analysis
• Real-time Data Processing with Intelligent Caching
• Multiple Interface Options (Chatbot, Dashboard, API)
• Advanced Risk Assessment and Confidence Scoring
• Modular Design for Easy Maintenance and Scaling
• Educational Value for Market Understanding"""
    
    ax.text(6, 2.75, features_text, ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_agent_interaction_diagram():
    """Create a detailed agent interaction diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Agent definitions with better positioning
    agents = [
        # (x, y, width, height, name, description, color)
        (1, 9, 2.5, 1.2, 'Data Collector', 'Gathers comprehensive stock and market data from multiple sources', '#E3F2FD'),
        (5, 9, 2.5, 1.2, 'Technical Analyzer', 'Calculates 20+ technical indicators and identifies patterns', '#E8F5E8'),
        (9, 9, 2.5, 1.2, 'Sentiment Analyzer', 'Analyzes news, social media, and market sentiment', '#F3E5F5'),
        (1, 6.5, 2.5, 1.2, 'Sentiment Integration', 'Combines technical and sentiment signals intelligently', '#FFF3E0'),
        (5, 6.5, 2.5, 1.2, 'Prediction Agent', 'LLM-powered final predictions with detailed reasoning', '#E0F2F1'),
        (9, 6.5, 2.5, 1.2, 'Risk Assessment', 'Evaluates market, volatility, and sentiment risks', '#FCE4EC'),
    ]
    
    # Draw agent boxes with better styling
    for x, y, w, h, name, desc, color in agents:
        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=1.5)
        ax.add_patch(box)
        
        # Add agent name
        ax.text(x + w/2, y + h - 0.3, name, 
                ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Add description (wrapped)
        ax.text(x + w/2, y + h/2, desc, 
                ha='center', va='center', 
                fontsize=8, wrap=True)
    
    # Draw connections between agents with better flow
    connections = [
        ((2.25, 9.6), (2.25, 7.7)),      # Data Collector to Sentiment Integration
        ((6.25, 9.6), (6.25, 7.7)),      # Technical Analyzer to Sentiment Integration
        ((10.25, 9.6), (10.25, 7.7)),    # Sentiment Analyzer to Sentiment Integration
        ((2.25, 7.1), (5, 7.1)),         # Sentiment Integration to Prediction Agent
        ((7.5, 7.1), (9, 7.1)),          # Prediction Agent to Risk Assessment
        ((10.25, 7.1), (6.25, 7.1)),     # Risk Assessment back to Prediction Agent
    ]
    
    for start, end in connections:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=15, fc="black", lw=1.5)
        ax.add_patch(arrow)
    
    # Add data flow labels with better positioning
    flow_labels = [
        (2.25, 8.5, 'Structured Data'),
        (6.25, 8.5, 'Technical Analysis'),
        (10.25, 8.5, 'Sentiment Analysis'),
        (3.5, 7.1, 'Integrated Signals'),
        (7.5, 7.1, 'LLM Analysis'),
        (8.5, 7.1, 'Risk Factors'),
    ]
    
    for x, y, text in flow_labels:
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=9, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Add main title
    ax.text(7, 11.5, 'AGENT INTERACTION DIAGRAM', 
            ha='center', va='center', fontsize=16, fontweight='bold', color='darkred')
    
    # Add subtitle
    ax.text(7, 11.2, 'Multi-Agent System Architecture and Data Flow', 
            ha='center', va='center', fontsize=12, style='italic', color='darkblue')
    
    # Add detailed legend
    legend_elements = [
        patches.Patch(color='#E3F2FD', label='Data Collection'),
        patches.Patch(color='#E8F5E8', label='Technical Analysis'),
        patches.Patch(color='#F3E5F5', label='Sentiment Analysis'),
        patches.Patch(color='#FFF3E0', label='Integration'),
        patches.Patch(color='#E0F2F1', label='LLM Prediction'),
        patches.Patch(color='#FCE4EC', label='Risk Assessment'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
              fontsize=10, title='Agent Types', title_fontsize=11)
    
    # Add system capabilities box
    capabilities_box = FancyBboxPatch((0.5, 1), 13, 2, 
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightgreen',
                                     edgecolor='green',
                                     linewidth=2)
    ax.add_patch(capabilities_box)
    
    capabilities_text = """SYSTEM CAPABILITIES:
• Real-time Data Processing with Intelligent Caching
• Advanced Technical Analysis with 20+ Indicators
• Multi-source Sentiment Analysis (News, Social Media, Reddit)
• LLM-Powered Predictions with Detailed Reasoning
• Comprehensive Risk Assessment and Confidence Scoring
• Multiple Interface Options (Chatbot, Dashboard, API)
• Modular Architecture for Easy Maintenance and Scaling"""
    
    ax.text(7, 2, capabilities_text, ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_data_flow_diagram():
    """Create a detailed data flow diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define data flow components
    components = [
        # (x, y, width, height, name, description, color)
        (1, 8, 2, 1, 'User Input', 'Ticker Symbol\n(e.g., AAPL)', '#E3F2FD'),
        (4, 8, 2, 1, 'Data Collection', 'Yahoo Finance\nMarket APIs', '#F3E5F5'),
        (7, 8, 2, 1, 'Technical Analysis', 'Indicators\nPatterns\nSignals', '#E8F5E8'),
        (10, 8, 2, 1, 'Sentiment Analysis', 'News\nSocial Media\nReddit', '#FFF8E1'),
        (2.5, 5.5, 2, 1, 'Data Integration', 'Signal Alignment\nConfidence Adjustment', '#FFF3E0'),
        (6, 5.5, 2, 1, 'LLM Processing', 'Gemini/Groq\nAnalysis', '#E0F2F1'),
        (9.5, 5.5, 2, 1, 'Risk Assessment', 'Market Risk\nVolatility\nSentiment Risk', '#FCE4EC'),
        (1, 3, 2, 1, 'Chatbot', 'Conversational\nInterface', '#FCE4EC'),
        (4, 3, 2, 1, 'Dashboard', 'Visual\nCharts', '#FCE4EC'),
        (7, 3, 2, 1, 'API', 'Programmatic\nAccess', '#FCE4EC'),
        (10, 3, 2, 1, 'Export', 'Data\nExport', '#FCE4EC'),
    ]
    
    # Draw components
    for x, y, w, h, name, desc, color in components:
        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=1.5)
        ax.add_patch(box)
        
        # Add component name
        ax.text(x + w/2, y + h - 0.2, name, 
                ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # Add description
        ax.text(x + w/2, y + h/2, desc, 
                ha='center', va='center', 
                fontsize=8, wrap=True)
    
    # Draw data flow arrows
    flow_arrows = [
        ((2, 8.5), (2, 7.5)),      # User Input to Data Collection
        ((5, 8.5), (5, 7.5)),      # Data Collection to Technical Analysis
        ((8, 8.5), (8, 7.5)),      # Technical Analysis to Sentiment Analysis
        ((11, 8.5), (11, 7.5)),    # Sentiment Analysis to Data Integration
        ((3.5, 6.5), (3.5, 6)),    # Data Integration to LLM Processing
        ((7, 6.5), (7, 6)),        # LLM Processing to Risk Assessment
        ((10.5, 6.5), (10.5, 6)),  # Risk Assessment to Outputs
        ((2, 4), (2, 3.5)),        # Risk Assessment to Chatbot
        ((5, 4), (5, 3.5)),        # Risk Assessment to Dashboard
        ((8, 4), (8, 3.5)),        # Risk Assessment to API
        ((11, 4), (11, 3.5)),      # Risk Assessment to Export
    ]
    
    for start, end in flow_arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=15, fc="black", lw=1.5)
        ax.add_patch(arrow)
    
    # Add title
    ax.text(7, 9.5, 'DATA FLOW ARCHITECTURE', 
            ha='center', va='center', fontsize=16, fontweight='bold', color='darkred')
    
    # Add subtitle
    ax.text(7, 9.2, 'End-to-End Data Processing Pipeline', 
            ha='center', va='center', fontsize=12, style='italic', color='darkblue')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create and save all diagrams
    print("🔄 Generating workflow diagrams...")
    
    # Main workflow diagram
    fig1 = create_main_workflow_diagram()
    fig1.savefig('docs/main_workflow_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ Main workflow diagram saved as 'docs/main_workflow_diagram.png'")
    
    # Agent interaction diagram
    fig2 = create_agent_interaction_diagram()
    fig2.savefig('docs/agent_interaction_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ Agent interaction diagram saved as 'docs/agent_interaction_diagram.png'")
    
    # Data flow diagram
    fig3 = create_data_flow_diagram()
    fig3.savefig('docs/data_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ Data flow diagram saved as 'docs/data_flow_diagram.png'")
    
    print("🎉 All diagrams generated successfully!")
    plt.show()
