---
title: Bitcoin Price Forecasting Portal
emoji: 📈
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# Bitcoin Price Forecasting Portal

An interactive Streamlit application for Bitcoin time-series analysis and forecasting using Kaggle-style BTC CSV files.

## Features

- CSV upload for BTC historical datasets
- Automatic timestamp parsing (supports common Kaggle columns like `Timestamp`, `Date`, `Datetime`)
- User-selectable price target (`Close`, `Open`, `High`, `Low`, or any numeric column)
- Data validation and preparation:
  - Chronological sorting check
  - Missing-day detection and filling for daily modeling
- Forecasting models:
  - ARIMA (auto-tuned via pmdarima)
  - Prophet
  - LSTM (Deep Learning)
  - Holt-Winters (Exponential Smoothing)
- Forecast controls from sidebar:
  - Forecast horizon (days)
  - Confidence interval (80-99%)
- Backtesting metrics (USD):
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
- Interactive Plotly chart with:
  - Historical prices
  - Projected trend
  - Uncertainty zone
  - Forecast start marker

## Dataset Used (Testing)

Kaggle Bitcoin Historical Data (minute OHLCV):

- https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data

Upload any BTC CSV via the app's "Upload CSV" option.

## How to Run Locally

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Streamlit:

```bash
streamlit run app.py
```

4. Open the local URL shown in terminal (typically `http://localhost:8501`).

## Project Structure

```text
app.py
requirements.txt
README.md
```

## How The Models Handle Crypto Volatility

- **ARIMA** captures autocorrelation and trend through differencing and lag structure. It can model short-term momentum/reversion patterns in BTC while providing confidence intervals from model uncertainty.
- **Prophet** models non-linear trend shifts and recurring seasonal effects and is robust to missing timestamps, which helps with noisy and irregular crypto market behavior.
- **LSTM (Deep Learning)** uses a recurrent neural network with memory cells to learn complex non-linear temporal patterns in price sequences, adapting to sudden regime changes common in crypto markets.
- **Holt-Winters** applies exponential smoothing with additive trend and seasonal components, capturing weekly price cycles while remaining lightweight and fast to train.
- All models are evaluated with holdout backtesting (MAE/RMSE in USD), so performance is measured on unseen historical periods before future projection.

## Notes

- The app resamples to **daily** data for consistent day-ahead forecasting and easier interpretation of forecast horizon in days.
- For best results, use datasets with long date coverage and clean OHLC columns.
