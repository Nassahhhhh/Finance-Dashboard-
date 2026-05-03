# 🛡️ Insight Horizon
### Institutional-Grade Portfolio Analytics Dashboard

A 5-year quantitative finance dashboard built with Python and Streamlit, using Random Forest Regression to capture and project market momentum.

---

## 📌 Project Info

| Field | Details |
|-------|---------|
| **Student** | Syed Hassan Ahmed |
| **Field** | Computational Finance |
| **Course** | Programming Fundamentals |
| **Instructor** | Sir Wasiq Noor |
| **University** | NED University of Engineering and Technology |

---

## 📊 Features

- **5-Year Cumulative Growth** — Tracks how $1 invested in each stock grows over 5 years
- **Sharpe Ratio Analysis** — Measures risk-adjusted return for each selected stock
- **Risk vs. Reward Map** — Scatter plot of annual volatility vs annual return
- **AI Price Forecast** — Random Forest model projecting future price patterns with realistic market noise
- **Interactive Sidebar** — Select up to 4 stocks and adjust forecast horizon from 1 to 6 months
- **Performance Summary Table** — Annual Return, Volatility, and Sharpe Ratio for all selected stocks

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | Interactive web dashboard |
| yFinance | Real-time stock data from Yahoo Finance |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical calculations |
| Matplotlib | Data visualization and charting |
| scikit-learn | Random Forest Regressor model |

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/your-username/insight-horizon.git
cd insight-horizon
```

**2. Install dependencies**
```bash
pip install streamlit yfinance pandas numpy matplotlib scikit-learn
```

**3. Run the app**
```bash
streamlit run final_dashboard.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 🌐 Live Demo

The app is deployed on Streamlit Cloud:
👉 **[Click here to open the dashboard](https://your-app-link.streamlit.app)**

> Replace the link above with your actual Streamlit Cloud URL

---

## 📁 Project Structure

```
insight-horizon/
│
├── final_dashboard.py      # Main application file
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📦 Requirements File

Create a `requirements.txt` with the following:
```
streamlit
yfinance
pandas
numpy
matplotlib
scikit-learn
```

---

## 📈 How It Works

1. **Data Collection** — Fetches 5 years of daily closing prices via yFinance
2. **Data Cleaning** — Forward/backward fills missing values, drops nulls
3. **Metric Calculation** — Computes annualised return, volatility, and Sharpe Ratio
4. **ML Forecasting** — Trains Random Forest on historical prices, adds Gaussian noise for realism
5. **Visualization** — Renders 4 interactive Matplotlib charts inside Streamlit

---

## ⚠️ Disclaimer

This dashboard is built for **educational purposes only**. The AI forecast is a pattern-based projection and should **not** be used as financial investment advice.

---

*Built with ❤️ for Programming Fundamentals — NED University of Engineering and Technology*
