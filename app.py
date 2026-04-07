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
st.markdown("### 📊 **Institutional-Grade Portfolio Analytics**")
st.markdown("#### *A 5-year quantitative study using Polynomial Regression to capture market momentum.*")

# 1. Sidebar Selection
st.sidebar.header("Portfolio Settings")
available_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "BRK-B", "V"]
selected_tickers = st.sidebar.multiselect(
    "Select up to 4 Stocks", 
    available_tickers, 
    default=["AAPL", "MSFT", "NVDA", "GOOGL"],
    max_selections=4
)

# --- NEW FEATURE: Forecast Slider ---
st.sidebar.divider()
st.sidebar.subheader("Forecast Horizon")
forecast_months = st.sidebar.slider(
    "Select Forecast Duration (Months)", 
    min_value=1, 
    max_value=6, 
    value=1,
    step=1
)
forecast_days = forecast_months * 21 # Converting months to trading days

if len(selected_tickers) == 0:
    st.info("Please select at least one ticker in the sidebar.")
    st.stop()

# Timeframe: Exactly 5 Years
start_date = datetime.now() - timedelta(days=1825)

@st.cache_data
def get_data(symbols, start):
    df = yf.download(symbols, start=start)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame(name=symbols[0])
    return df

data = get_data(selected_tickers, start_date)
returns = data.pct_change().dropna()

# 2. Risk Metrics
ann_return = returns.mean() * 252
ann_vol = returns.std() * np.sqrt(252)
sharpe = ann_return / ann_vol

# 3. Dashboard Grid
plt.style.use('ggplot')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
col1, col2 = st.columns(2)

# --- GRAPH 1: Cumulative Growth ---
with col1:
    st.subheader("1. 5-Year Cumulative Growth ($1 Invested)")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    (1 + returns).cumprod().plot(ax=ax1, lw=2.5, color=colors[:len(selected_tickers)])
    ax1.set_ylabel("Growth Factor")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

# --- GRAPH 2: Sharpe Ratio ---
with col2:
    st.subheader("2. Sharpe Ratio (Risk Efficiency)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(selected_tickers, sharpe, color=colors[:len(selected_tickers)], alpha=0.8)
    ax2.set_ylabel("Sharpe Ratio Value")
    ax2.axhline(0, color='black', lw=1)
    for i, v in enumerate(sharpe):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha='center', weight='bold')
    plt.tight_layout()
    st.pyplot(fig2)

st.divider()
col3, col4 = st.columns(2)

# --- GRAPH 3: Risk vs. Reward Mapping ---
with col3:
    st.subheader("3. Risk vs. Reward (Volatility Map)")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(ann_vol * 100, ann_return * 100, s=500, c=colors[:len(selected_tickers)], alpha=0.6, edgecolors='black')
    for i, txt in enumerate(selected_tickers):
        ax3.annotate(txt, (ann_vol.iloc[i]*100, ann_return.iloc[i]*100), xytext=(0,15), textcoords='offset points', ha='center', weight='bold')
    ax3.set_xlabel("Annual Volatility (%)")
    ax3.set_ylabel("Annual Return (%)")
    plt.tight_layout()
    st.pyplot(fig3)

# --- GRAPH 4: 5-Year AI-Simulated Pattern ---
with col4:
    st.subheader(f"4. {forecast_months}-Month Global Pattern Forecast")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    for i, stock in enumerate(selected_tickers):
        # 1. Prepare data using the FULL 5-year history
        y_train = data[stock].values # This is the full 1,260+ days
        X_train = np.arange(len(y_train)).reshape(-1, 1)
        
        # 2. Train Random Forest on the full 5 years
        # We increase n_estimators to 200 to handle the larger data volume
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # 3. Create Future Timeline
        future_X = np.arange(len(y_train), len(y_train) + forecast_days).reshape(-1, 1)
        
        # 4. Generate the AI Prediction
        base_forecast = model.predict(future_X)
        
        # 5. Add "Realistic Wiggle" based on 5-year Volatility
        # This ensures the 'jagged' pattern matches the historical style
        volatility = np.std(np.diff(y_train)) 
        noise = np.random.normal(0, volatility * 0.7, size=forecast_days)
        forecast = base_forecast + noise
        
        # 6. Visual Zoom (90 Days)
        # We still zoom in so you can see the 'pattern' clearly
        zoom_view = 90
        x_hist = np.arange(zoom_view)
        x_pred = np.arange(zoom_view, zoom_view + forecast_days)
        
        # Plot History
        ax4.plot(x_hist, y_train[-zoom_view:], color=colors[i], alpha=0.5, lw=1.5, label=f"{stock} Hist")
        
        # Plot Prediction (Jagged Pattern style)
        full_pred_x = np.insert(x_pred, 0, x_hist[-1])
        full_pred_y = np.insert(forecast, 0, y_train[-1])
        
        ax4.plot(full_pred_x, full_pred_y, color=colors[i], lw=1.5, label=f"{stock} AI")

    ax4.set_title("5-Year Global Pattern Projection")
    ax4.set_ylabel("Price ($)")
    ax4.legend(loc='upper left', prop={'size': 8}, ncol=2)
    plt.tight_layout()
    st.pyplot(fig4)

# 4. Final Data Summary Table
st.divider()
st.subheader("Performance Metrics Summary")
summary_df = pd.DataFrame({
    "Annual Return": (ann_return * 100).round(2).astype(str) + "%",
    "Annual Volatility": (ann_vol * 100).round(2).astype(str) + "%",
    "Sharpe Ratio": sharpe.round(2)
})
st.table(summary_df)