import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("🛡️ Insight Horizon")
st.markdown("### 📊 **Institutional-Grade Portfolio Analytics**")
st.markdown("#### *A 5-year quantitative study using Forest Regression to capture market momentum.*")

st.sidebar.header("Portfolio Settings")
available_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "BRK-B", "V"]
selected_tickers = st.sidebar.multiselect(
    "Select up to 4 Stocks", 
    available_tickers, 
    default=["AAPL", "MSFT", "NVDA", "GOOGL"],
    max_selections=4
)

st.sidebar.divider()
st.sidebar.subheader("Forecast Horizon")
forecast_months = st.sidebar.slider(
    "Select Forecast Duration (Months)", 
    min_value=1, max_value=6, value=1, step=1
)
forecast_days = forecast_months * 21

if len(selected_tickers) == 0:
    st.info("Please select at least one ticker in the sidebar.")
    st.stop()

start_date = datetime.now() - timedelta(days=1825)

@st.cache_data
def get_data(symbols, start):
    df = yf.download(symbols, start=start)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame(name=symbols[0])
    return df[symbols]  # FIX: always enforce selected order

data = get_data(selected_tickers, start_date)
returns = data.pct_change().dropna()

ann_return = returns.mean() * 252
ann_vol = returns.std() * np.sqrt(252)
sharpe = (ann_return / ann_vol).reindex(selected_tickers)

plt.style.use('ggplot')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. 5-Year Cumulative Growth ($1 Invested)")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    (1 + returns).cumprod().plot(ax=ax1, lw=2.5, color=colors[:len(selected_tickers)])
    ax1.set_ylabel("Growth Factor")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

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

with col3:
    st.subheader("3. Risk vs. Reward (Volatility Map)")
    plot_vol = ann_vol.reindex(selected_tickers) * 100
    plot_ret = ann_return.reindex(selected_tickers) * 100
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(plot_vol.values, plot_ret.values, s=500,
                c=colors[:len(selected_tickers)], alpha=0.6, edgecolors='black')
    for i, txt in enumerate(selected_tickers):
        ax3.annotate(txt, (plot_vol.iloc[i], plot_ret.iloc[i]),
                     xytext=(0, 15), textcoords='offset points', ha='center', weight='bold')
    ax3.set_xlabel("Annual Volatility (%)")
    ax3.set_ylabel("Annual Return (%)")
    ax3.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig3)

with col4:
    st.subheader(f"4. {forecast_months}-Month Global Pattern Forecast")
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    for i, stock in enumerate(selected_tickers):
        y_train = data[stock].ffill().bfill().values
        X_train = np.arange(len(y_train)).reshape(-1, 1)

        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        future_X = np.arange(len(y_train), len(y_train) + forecast_days).reshape(-1, 1)
        base_forecast = model.predict(future_X)

        volatility = np.std(np.diff(y_train))
        noise = np.random.normal(0, volatility * 0.7, size=forecast_days)
        forecast = base_forecast + noise

        zoom_view = 90
        x_hist = np.arange(zoom_view)
        x_pred = np.arange(zoom_view, zoom_view + forecast_days)

        ax4.plot(x_hist, y_train[-zoom_view:], color=colors[i], alpha=0.5, lw=1.5, label=f"{stock} Hist")

        full_pred_x = np.insert(x_pred, 0, x_hist[-1])
        full_pred_y = np.insert(forecast, 0, y_train[-1])
        ax4.plot(full_pred_x, full_pred_y, color=colors[i], lw=1.5, label=f"{stock} AI")

    ax4.set_title("5-Year Global Pattern Projection")
    ax4.set_ylabel("Price ($)")
    ax4.legend(loc='upper left', prop={'size': 8}, ncol=2)
    plt.tight_layout()
    st.pyplot(fig4)

st.divider()
st.subheader("Performance Metrics Summary")
summary_df = pd.DataFrame({
    "Annual Return": (ann_return * 100).round(2).astype(str) + "%",
    "Annual Volatility": (ann_vol * 100).round(2).astype(str) + "%",
    "Sharpe Ratio": sharpe.round(2)
})
st.table(summary_df)