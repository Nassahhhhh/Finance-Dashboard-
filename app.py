import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# Page Configuration
st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("🛡️ 5-Year Risk Analysis & AI Forecast")

# --- Project Description ---
st.markdown("""
This dashboard provides a **comprehensive risk-reward analysis** of your selected portfolio. 
It calculates efficiency via the **Sharpe Ratio**, visualizes volatility, and uses a 
**Random Forest Regressor** to project potential price patterns over the coming months.
""")

# 1. Sidebar Selection
st.sidebar.header("Portfolio Settings")
available_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "BRK-B", "V"]
selected_tickers = st.sidebar.multiselect(
    "Select up to 4 Stocks", 
    available_tickers, 
    default=["AAPL", "MSFT", "NVDA", "GOOGL"],
    max_selections=4
)

st.sidebar.subheader("Forecast Horizon")
forecast_months = st.sidebar.slider("Months", 1, 6, 1)
forecast_days = forecast_months * 21 # Approx trading days

if not selected_tickers:
    st.info("Select tickers to begin.")
    st.stop()

# Timeframe
start_date = datetime.now() - timedelta(days=1825)

@st.cache_data
def get_data(symbols, start):
    df = yf.download(symbols, start=start)['Close']
    if len(symbols) == 1:
        df = df.to_frame(name=symbols[0])
    return df[symbols]

data = get_data(selected_tickers, start_date)
returns = data.pct_change().dropna()

# 2. Risk Metrics
ann_return = returns.mean() * 252
ann_vol = returns.std() * np.sqrt(252)
sharpe = ann_return / ann_vol

plt.style.use('ggplot')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 5-Year Cumulative Growth")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    cumulative = (1 + returns).cumprod()
    for i, stock in enumerate(selected_tickers):
        ax1.plot(cumulative.index, cumulative[stock], label=stock, color=colors[i], lw=2)
    ax1.set_ylabel("Growth Multiplier")
    ax1.set_xlabel("Date")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.subheader("2. Sharpe Ratio (Risk Efficiency)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(sharpe.index, sharpe.values, color=colors[:len(selected_tickers)], alpha=0.8)
    for i, v in enumerate(sharpe.values):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', weight='bold')
    ax2.set_ylabel("Sharpe Ratio Value")
    ax2.set_xlabel("Ticker")
    st.pyplot(fig2)

st.divider()
col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Risk vs. Reward (Volatility Analysis)")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(ann_vol * 100, ann_return * 100, s=400, c=colors[:len(selected_tickers)], alpha=0.6)
    for i, txt in enumerate(selected_tickers):
        ax3.annotate(txt, (ann_vol.iloc[i]*100, ann_return.iloc[i]*100), xytext=(0,15), textcoords='offset points', ha='center')
    ax3.set_xlabel("Annualized Volatility (%)")
    ax3.set_ylabel("Annualized Return (%)")
    st.pyplot(fig3)

with col4:
    st.subheader("4. AI Pattern Forecast")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    for i, stock in enumerate(selected_tickers):
        prices = data[stock].dropna().values
        X = np.arange(len(prices)).reshape(-1, 1)
        
        # Training the AI model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, prices)

        # Future Indices
        future_X = np.arange(len(prices), len(prices) + forecast_days).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # --- FIX: Connecting and Smoothing the line ---
        # We show the last 30 days of history
        hist_view = prices[-30:]
        x_hist = np.arange(len(hist_view))
        
        # Plot History
        ax4.plot(x_hist, hist_view, color=colors[i], alpha=0.5, lw=1)
        
        # Connect Forecast to the last actual point
        x_fore = np.arange(len(hist_view) - 1, len(hist_view) + forecast_days - 1)
        # Ensure the first point of forecast is the last point of history
        y_fore = np.concatenate([[hist_view[-1]], forecast])
        
        ax4.plot(x_fore, y_fore, color=colors[i], lw=2, linestyle='--', label=f"{stock} Forecast")

    ax4.set_title(f"Next {forecast_days} Trading Days Pattern", fontsize=12)
    ax4.set_xlabel("Relative Trading Days")
    ax4.set_ylabel("Stock Price ($)")
    ax4.legend(ncol=2, fontsize='small')
    st.pyplot(fig4)

st.divider()
st.subheader("Performance Summary Table")
st.table(pd.DataFrame({
    "Return": (ann_return * 100).round(2).astype(str) + '%',
    "Risk (Vol)": (ann_vol * 100).round(2).astype(str) + '%',
    "Sharpe": sharpe.round(2)
}, index=selected_tickers))
