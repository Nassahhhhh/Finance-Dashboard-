import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# Page Configuration
st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("🛡️ 5-Year Risk Analysis & AI Forest Forecast")

st.markdown("""
This dashboard analyzes historical performance and uses a **Random Forest Regressor** to simulate potential future price patterns based on past volatility.
""")

# 1. Sidebar Selection
st.sidebar.header("Portfolio Settings")
available_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "BRK-B", "V"]
selected_tickers = st.sidebar.multiselect(
    "Select Stocks", 
    available_tickers, 
    default=["AAPL", "MSFT", "NVDA", "GOOGL"],
    max_selections=4
)

st.sidebar.subheader("Forecast Horizon")
forecast_months = st.sidebar.slider("Months", 1, 6, 3)
forecast_days = forecast_months * 21 

if not selected_tickers:
    st.info("Select tickers to begin.")
    st.stop()

# Data Fetching
start_date = datetime.now() - timedelta(days=1825)

@st.cache_data
def get_data(symbols, start):
    df = yf.download(symbols, start=start)['Close']
    if len(symbols) == 1:
        df = df.to_frame(name=symbols[0])
    return df[symbols]

data = get_data(selected_tickers, start_date)
returns = data.pct_change().dropna()

# Calculations
ann_return = returns.mean() * 252
ann_vol = returns.std() * np.sqrt(252)
sharpe = ann_return / ann_vol

plt.style.use('ggplot')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# --- Row 1: Cumulative & Sharpe ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("1. 5-Year Cumulative Growth")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    (1 + returns).cumprod().plot(ax=ax1, color=colors[:len(selected_tickers)])
    ax1.set_ylabel("Growth Multiplier")
    ax1.set_xlabel("Date")
    st.pyplot(fig1)

with col2:
    st.subheader("2. Sharpe Ratio (Efficiency)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(sharpe.index, sharpe.values, color=colors[:len(selected_tickers)])
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_xlabel("Ticker")
    st.pyplot(fig2)

st.divider()

# --- Row 2: Risk/Reward & FIXED Forest Forecast ---
col3, col4 = st.columns(2)
with col3:
    st.subheader("3. Risk vs. Reward")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(ann_vol * 100, ann_return * 100, s=300, c=colors[:len(selected_tickers)], alpha=0.7)
    for i, txt in enumerate(selected_tickers):
        ax3.annotate(txt, (ann_vol.iloc[i]*100, ann_return.iloc[i]*100), xytext=(5,5), textcoords='offset points')
    ax3.set_xlabel("Volatility (%)")
    ax3.set_ylabel("Return (%)")
    st.pyplot(fig3)

with col4:
    st.subheader("4. AI Pattern Forecast (Random Forest)")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    for i, stock in enumerate(selected_tickers):
        prices = data[stock].dropna().values
        X_train = np.arange(len(prices)).reshape(-1, 1)
        
        # Forest Model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, prices)

        # Generate Forecast
        future_X = np.arange(len(prices), len(prices) + forecast_days).reshape(-1, 1)
        forecast_base = rf.predict(future_X)
        
        # Add a bit of random movement so it's not a flat line
        volatility = np.std(np.diff(prices[-20:])) 
        noise = np.random.normal(0, volatility, size=forecast_days)
        forecast_final = forecast_base + noise

        # Plotting Setup
        hist_segment = prices[-60:]
        x_hist = np.arange(len(hist_segment))
        
        # x_fore starts at the last index of x_hist
        x_fore = np.arange(len(hist_segment) - 1, len(hist_segment) + forecast_days - 1)
        
        # y_fore combines the last historical point with the forecast
        y_fore = np.concatenate([[hist_segment[-1]], forecast_final])
        
        # Plotting
        ax4.plot(x_hist, hist_segment, color=colors[i], alpha=0.3)
        ax4.plot(x_fore, y_fore, color=colors[i], lw=2, linestyle='--', label=f"{stock} (RF)")

    ax4.set_xlabel("Trading Days (Context + Future)")
    ax4.set_ylabel("Price ($)")
    ax4.legend(ncol=2, fontsize='x-small')
    st.pyplot(fig4)

st.divider()
st.table(pd.DataFrame({
    "Return": (ann_return * 100).round(2).astype(str) + '%',
    "Risk": (ann_vol * 100).round(2).astype(str) + '%',
    "Sharpe": sharpe.round(2)
}, index=selected_tickers))
