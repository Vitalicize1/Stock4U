import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Stock4U - AI Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_stock_data(ticker, period="1mo"):
    """Get stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data, stock.info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

def calculate_technical_indicators(data):
    """Calculate basic technical indicators"""
    if data is None or data.empty:
        return None
    
    # Simple Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

def create_candlestick_chart(data, ticker):
    """Create interactive candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=ticker
    )])
    
    # Add moving averages
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_20'],
            mode='lines', name='SMA 20',
            line=dict(color='orange', width=2)
        ))
    
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['SMA_50'],
            mode='lines', name='SMA 50',
            line=dict(color='blue', width=2)
        ))
    
    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_white',
        height=500
    )
    
    return fig

def generate_prediction(data, ticker):
    """Generate a simple prediction based on technical analysis"""
    if data is None or data.empty:
        return None, None, None
    
    current_price = data['Close'].iloc[-1]
    
    # Simple trend analysis
    recent_trend = data['Close'].tail(5).pct_change().mean()
    
    # RSI analysis
    current_rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
    
    # Moving average analysis
    sma_20 = data['SMA_20'].iloc[-1] if 'SMA_20' in data.columns else current_price
    sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns else current_price
    
    # Simple prediction logic
    prediction_score = 0
    
    # Trend factor
    if recent_trend > 0.01:  # 1% daily growth
        prediction_score += 1
    elif recent_trend < -0.01:
        prediction_score -= 1
    
    # RSI factor
    if current_rsi < 30:
        prediction_score += 1  # Oversold, potential bounce
    elif current_rsi > 70:
        prediction_score -= 1  # Overbought, potential drop
    
    # Moving average factor
    if current_price > sma_20 > sma_50:
        prediction_score += 1  # Bullish alignment
    elif current_price < sma_20 < sma_50:
        prediction_score -= 1  # Bearish alignment
    
    # Generate prediction
    if prediction_score >= 1:
        prediction = "BULLISH 📈"
        confidence = min(85, 60 + abs(prediction_score) * 10)
        risk_level = "LOW" if prediction_score >= 2 else "MEDIUM"
    elif prediction_score <= -1:
        prediction = "BEARISH 📉"
        confidence = min(85, 60 + abs(prediction_score) * 10)
        risk_level = "LOW" if prediction_score <= -2 else "MEDIUM"
    else:
        prediction = "NEUTRAL ➡️"
        confidence = 50
        risk_level = "MEDIUM"
    
    return prediction, confidence, risk_level

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock4U - AI Stock Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Stock input
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
    
    # Time period
    period_options = {
        "1 Week": "1wk",
        "1 Month": "1mo", 
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y"
    }
    period = st.sidebar.selectbox("Time Period", list(period_options.keys()), index=1)
    period_value = period_options[period]
    
    # Analysis button
    analyze_button = st.sidebar.button("🚀 Analyze Stock", type="primary")
    
    # Main content
    if analyze_button and ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            # Get stock data
            data, info = get_stock_data(ticker, period_value)
            
            if data is not None and not data.empty:
                # Calculate indicators
                data = calculate_technical_indicators(data)
                
                # Display current price and basic info
                col1, col2, col3, col4 = st.columns(4)
                
                current_price = data['Close'].iloc[-1]
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    st.metric("Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
                
                with col3:
                    if info and 'volume' in info:
                        st.metric("Volume", f"{info['volume']:,}")
                    else:
                        st.metric("Volume", "N/A")
                
                with col4:
                    if info and 'marketCap' in info:
                        market_cap = info['marketCap'] / 1e9  # Convert to billions
                        st.metric("Market Cap", f"${market_cap:.1f}B")
                    else:
                        st.metric("Market Cap", "N/A")
                
                # Chart
                st.subheader("📊 Price Chart")
                fig = create_candlestick_chart(data, ticker)
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction
                st.subheader("🤖 AI Prediction")
                prediction, confidence, risk_level = generate_prediction(data, ticker)
                
                if prediction:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>Prediction</h3>
                            <h2>{prediction}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Confidence</h3>
                            <h2>{confidence}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Risk Level</h3>
                            <h2>{risk_level}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Technical Indicators
                st.subheader("📈 Technical Indicators")
                
                if 'RSI' in data.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        current_rsi = data['RSI'].iloc[-1]
                        st.metric("RSI (14)", f"{current_rsi:.1f}")
                        
                        if current_rsi > 70:
                            st.warning("⚠️ Overbought - Consider selling")
                        elif current_rsi < 30:
                            st.success("✅ Oversold - Consider buying")
                    
                    with col2:
                        if 'MACD' in data.columns:
                            current_macd = data['MACD'].iloc[-1]
                            signal = data['Signal'].iloc[-1]
                            st.metric("MACD", f"{current_macd:.4f}")
                            st.metric("Signal", f"{signal:.4f}")
                            
                            if current_macd > signal:
                                st.success("✅ MACD above signal - Bullish")
                            else:
                                st.error("❌ MACD below signal - Bearish")
                
                # Recent performance
                st.subheader("📊 Recent Performance")
                
                # Calculate performance metrics
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_return = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                    st.metric("Total Return", f"{total_return:.2f}%")
                
                with col2:
                    st.metric("Volatility (Annual)", f"{volatility:.2f}%")
                
                with col3:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                # Disclaimer
                st.markdown("""
                <div class="warning-box">
                    <h4>⚠️ Disclaimer</h4>
                    <p>This analysis is for educational purposes only and should not be considered as financial advice. 
                    Always do your own research and consult with a financial advisor before making investment decisions.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol and try again.")
    
    # Default view when no analysis is run
    else:
        st.markdown("""
        ## 🎯 Welcome to Stock4U!
        
        **Stock4U** is an AI-powered stock analysis tool that provides:
        
        - 📊 **Real-time stock data** from Yahoo Finance
        - 🤖 **AI-powered predictions** based on technical analysis
        - 📈 **Interactive charts** with technical indicators
        - ⚡ **Quick analysis** for any stock ticker
        
        ### 🚀 How to use:
        1. Enter a stock ticker (e.g., AAPL, MSFT, GOOGL)
        2. Select a time period
        3. Click "Analyze Stock"
        4. View predictions, charts, and technical indicators
        
        ### 💡 Pro Tips:
        - Use popular tickers like AAPL, MSFT, TSLA, GOOGL
        - Try different time periods for different insights
        - Check the technical indicators for confirmation
        
        ---
        
        **Ready to analyze? Use the sidebar to get started! 🚀**
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📈 Stock4U - AI Stock Analysis | Built with Streamlit</p>
        <p>For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
