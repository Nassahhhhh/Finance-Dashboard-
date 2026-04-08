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
    st.subheader(f"4. AI Pattern Forecast ({forecast_months}M)")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    for i, stock in enumerate(selected_tickers):
        y_train = data[stock].dropna().values
        X_train = np.arange(len(y_train)).reshape(-1, 1)
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)

        # Forecasting
        future_X = np.arange(len(y_train), len(y_train) + forecast_days).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # --- FIX: Connecting the lines ---
        # We append the last actual price to the start of the forecast to close the gap
        hist_segment = y_train[-60:] # Show last 60 days for context
        x_hist = np.arange(len(hist_segment))
        
        # Plotting History
        ax4.plot(x_hist, hist_segment, color=colors[i], alpha=0.4, label=f"{stock} Hist" if i==0 else "")
        
        # Plotting Forecast (Starts exactly at the end of history)
        x_fore = np.arange(len(hist_segment) - 1, len(hist_segment) + forecast_days - 1)
        fore_plot_data = np.insert(forecast, 0, hist_segment[-1])
        ax4.plot(x_fore, fore_plot_data, color=colors[i], lw=2, linestyle='--', label=f"{stock} Forecast")

    ax4.set_xlabel("Trading Days (Recent + Future)")
    ax4.set_ylabel("Stock Price ($)")
    ax4.legend(loc='upper left', fontsize='small', ncol=2)
    st.pyplot(fig4)

st.divider()
st.subheader("📊 Performance Metrics Summary")
summary_df = pd.DataFrame({
    "Annual Return": (ann_return * 100).round(2).astype(str) + '%',
    "Annual Volatility": (ann_vol * 100).round(2).astype(str) + '%',
    "Sharpe Ratio": sharpe.round(3)
}, index=selected_tickers)
st.table(summary_df)
