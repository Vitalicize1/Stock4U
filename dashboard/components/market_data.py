import streamlit as st
import yfinance as yf
import plotly.graph_objects as go


def display_market_data(ticker: str) -> None:
    """Display current market data for the ticker."""

    try:
        # Fetch current market data
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="5d")

        if hist.empty:
            st.error("Unable to fetch market data")
            return

        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current Price",
                f"${latest['Close']:.2f}",
                f"{latest['Close'] - prev['Close']:.2f}"
            )

        with col2:
            change_pct = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
            st.metric(
                "Daily Change",
                f"{change_pct:.2f}%",
                f"{latest['Close'] - prev['Close']:.2f}"
            )

        with col3:
            st.metric("Volume", f"{latest['Volume']:,}")

        with col4:
            st.metric("Market Cap", f"${info.get('marketCap', 0):,.0f}")

        # Create price chart
        st.subheader("📈 Price Chart (Last 5 Days)")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='OHLC'
        ))

        fig.update_layout(
            title=f"{ticker} Stock Price",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")


