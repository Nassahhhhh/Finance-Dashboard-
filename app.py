import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# Page Configuration
st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("🛡️ 5-Year Risk Analysis & Curved Trend Prediction")

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
forecast_days = forecast_months * 21

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
    # FORCE the columns to stay in the order you selected them
    return df[symbols]

data = get_data(selected_tickers, start_date)
returns = data.pct_change().dropna()

# 2. Risk Metrics (Calculated once, correctly indexed)
ann_return = returns.mean() * 252
ann_vol = returns.std() * np.sqrt(252)
sharpe = ann_return / ann_vol

plt.style.use('ggplot')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 5-Year Cumulative Growth")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    (1 + returns).cumprod().plot(ax=ax1, lw=2.5, color=colors[:len(selected_tickers)])
    st.pyplot(fig1)

with col2:
    st.subheader("2. Sharpe Ratio (Risk Efficiency)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # Using the explicit index from the 'sharpe' series to ensure labels match bars
    ax2.bar(sharpe.index, sharpe.values, color=colors[:len(selected_tickers)], alpha=0.8)
    for i, v in enumerate(sharpe.values):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', weight='bold')
    st.pyplot(fig2)

st.divider()
col3, col4 = st.columns(2)

with col3:
    st.subheader("3. Risk vs. Reward")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(ann_vol * 100, ann_return * 100, s=500, c=colors[:len(selected_tickers)], alpha=0.6)
    for i, txt in enumerate(sharpe.index):
        ax3.annotate(txt, (ann_vol.iloc[i]*100, ann_return.iloc[i]*100), xytext=(0,15), textcoords='offset points', ha='center')
    st.pyplot(fig3)

with col4:
    st.subheader(f"4. AI Pattern Forecast")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    for i, stock in enumerate(selected_tickers):
        y_train = data[stock].dropna().values
        X_train = np.arange(len(y_train)).reshape(-1, 1)
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)
        
        future_X = np.arange(len(y_train), len(y_train) + forecast_days).reshape(-1, 1)
        forecast = model.predict(future_X) + np.random.normal(0, np.std(np.diff(y_train)) * 0.7, size=forecast_days)
        
        # Zoom view
        x_hist = np.arange(90)
        ax4.plot(x_hist, y_train[-90:], color=colors[i], alpha=0.4)
        ax4.plot(np.arange(90, 90 + forecast_days), forecast, color=colors[i], lw=2, linestyle='--')
    st.pyplot(fig4)

st.divider()
st.subheader("Performance Metrics Summary")
# Ensure the table uses the exact same order as the charts
summary_df = pd.DataFrame({
    "Annual Return": (ann_return * 100).round(2).map("{:.2f}%".format),
    "Annual Volatility": (ann_vol * 100).round(2).map("{:.2f}%".format),
    "Sharpe Ratio": sharpe.round(4)
}, index=sharpe.index)
st.table(summary_df)
