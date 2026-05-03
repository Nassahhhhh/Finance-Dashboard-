import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Page Configuration
st.set_page_config(page_title="Finance Dashboard", layout="wide")
st.title("🛡️ Insight Horizon")
st.markdown("### 📊 **Institutional-Grade Portfolio Analytics**")
st.markdown("#### *A 5-year quantitative study using Forest Regression to capture market momentum.*")

# 1. Sidebar Selection
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
    min_value=1,
    max_value=6,
    value=1,
    step=1
)
forecast_days = forecast_months * 21  # Trading days

if len(selected_tickers) == 0:
    st.info("Please select at least one ticker in the sidebar.")
    st.stop()

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
sharpe = sharpe.reindex(selected_tickers)

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

# --- GRAPH 3: Risk vs. Reward ---
with col3:
    st.subheader("3. Risk vs. Reward (Volatility Map)")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(ann_vol * 100, ann_return * 100, s=500, c=colors[:len(selected_tickers)], alpha=0.6, edgecolors='black')
    for i, txt in enumerate(selected_tickers):
        ax3.annotate(txt, (ann_vol.iloc[i]*100, ann_return.iloc[i]*100),
                     xytext=(0, 15), textcoords='offset points', ha='center', weight='bold')
    ax3.set_xlabel("Annual Volatility (%)")
    ax3.set_ylabel("Annual Return (%)")
    plt.tight_layout()
    st.pyplot(fig3)

# ─────────────────────────────────────────────────────────────────────────────
# --- GRAPH 4: IMPROVED Random Forest Forecast with Lag Features + Train/Test Split ---
# ─────────────────────────────────────────────────────────────────────────────
with col4:
    st.subheader(f"4. {forecast_months}-Month AI Forecast (Random Forest)")
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    for i, stock in enumerate(selected_tickers):

        # ── STEP 1: Get price series ──────────────────────────────────────
        prices = data[stock].ffill().bfill()

        # ── STEP 2: Engineer Lag Features ────────────────────────────────
        # Instead of using a raw index, we create meaningful features:
        # - Lag-1  : yesterday's price
        # - Lag-5  : price 1 week ago
        # - Lag-21 : price 1 month ago
        # - Rolling mean (21-day): short-term average price trend
        df_feat = pd.DataFrame({'price': prices})
        df_feat['lag1']     = df_feat['price'].shift(1)
        df_feat['lag5']     = df_feat['price'].shift(5)
        df_feat['lag21']    = df_feat['price'].shift(21)
        df_feat['roll21']   = df_feat['price'].rolling(21).mean()
        df_feat.dropna(inplace=True)

        X = df_feat[['lag1', 'lag5', 'lag21', 'roll21']].values
        y = df_feat['price'].values

        # ── STEP 3: Train / Test Split (80% train, 20% test) ─────────────
        split = int(len(X) * 0.80)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # ── STEP 4: Train Random Forest ───────────────────────────────────
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        # ── STEP 5: Evaluate on Test Set (RMSE) ───────────────────────────
        y_pred_test = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # ── STEP 6: Rolling Forecast for Future Days ──────────────────────
        # Seed with the last known lag values from the full dataset
        last_prices = list(prices.values[-22:])  # keep a buffer of 22 prices
        forecast = []

        for _ in range(forecast_days):
            lag1  = last_prices[-1]
            lag5  = last_prices[-5]  if len(last_prices) >= 5  else last_prices[0]
            lag21 = last_prices[-21] if len(last_prices) >= 21 else last_prices[0]
            roll21 = np.mean(last_prices[-21:])

            next_price = model.predict([[lag1, lag5, lag21, roll21]])[0]

            # Add calibrated noise based on historical volatility
            volatility = np.std(np.diff(prices.values))
            next_price += np.random.normal(0, volatility * 0.5)

            forecast.append(next_price)
            last_prices.append(next_price)

        # ── STEP 7: Plot ──────────────────────────────────────────────────
        zoom_view = 90
        hist_prices = prices.values[-zoom_view:]

        x_hist = np.arange(zoom_view)
        x_pred = np.arange(zoom_view, zoom_view + forecast_days)

        # Historical line
        ax4.plot(x_hist, hist_prices, color=colors[i], alpha=0.5, lw=1.5,
                 label=f"{stock} Hist")

        # Forecast line (connect from last historical point)
        full_pred_x = np.insert(x_pred, 0, x_hist[-1])
        full_pred_y = np.insert(forecast, 0, hist_prices[-1])
        ax4.plot(full_pred_x, full_pred_y, color=colors[i], lw=1.8,
                 label=f"{stock} Forecast (RMSE: ${rmse:.2f})")

    ax4.axvline(x=zoom_view - 1, color='gray', linestyle='--', lw=1, alpha=0.6)
    ax4.text(zoom_view, ax4.get_ylim()[0], ' Forecast Start', color='gray', fontsize=8)
    ax4.set_title(f"{forecast_months}-Month Random Forest Price Forecast")
    ax4.set_ylabel("Price ($)")
    ax4.set_xlabel("Trading Days")
    ax4.legend(loc='upper left', prop={'size': 7}, ncol=1)
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
