"""
forecasting.py – All data-loading, preparation, and model logic.
Extracted from the original monolithic app.py to keep the UI layer clean.
Every function here is unchanged from the original working implementation.
"""

from pathlib import Path
from statistics import NormalDist
from io import BytesIO
import importlib.util
import re
import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ---------- Lazy-import tools to (avoid loading heavy libs at startup) ----------

def _import_arima_stack():
    """Import ARIMA + auto_arima on first use."""
    from statsmodels.tsa.arima.model import ARIMA
    from pmdarima import auto_arima
    return ARIMA, auto_arima


def _import_prophet():
    from prophet import Prophet
    return Prophet


def _import_holt_winters():
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    return ExponentialSmoothing


def _import_lstm_stack():
    """Import TensorFlow/Keras on first use with minimal overhead."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    return Sequential, LSTM, Dense, Dropout, EarlyStopping, MinMaxScaler


def is_lstm_available():
    """Return True when TensorFlow is installed in the active environment."""
    return importlib.util.find_spec("tensorflow") is not None

DEFAULT_DATASET_PATH = Path("./datasets/btcusd_1-min_data.csv")
KAGGLE_DATASET_REF = "mczielinski/bitcoin-historical-data"
KAGGLE_DATASET_FILE = "btcusd_1-min_data.csv"

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "date",
    "datetime",
    "time",
]

TIMESTAMP_PARSE_SAMPLE_SIZE = 300


def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_timestamp_column(columns):
    normalized_map = {col: normalize_name(col) for col in columns}
    for candidate in TIMESTAMP_CANDIDATES:
        for col, normalized in normalized_map.items():
            if normalized == candidate:
                return col
    for col, normalized in normalized_map.items():
        if "time" in normalized or "date" in normalized:
            return col
    return None


def get_timestamp_candidate_columns(df):
    candidates = []
    for col in df.columns:
        normalized = normalize_name(col)
        by_name = (
            "timestamp" in normalized
            or "datetime" in normalized
            or "date" in normalized
            or "time" in normalized
            or "unix" in normalized
        )

        if by_name:
            candidates.append(col)
            continue

        sample = df[col].dropna().head(TIMESTAMP_PARSE_SAMPLE_SIZE)
        if sample.empty:
            continue

        sample_as_str = sample.astype(str)
        has_datetime_separators = sample_as_str.str.contains(r"[-/:Tt ]", regex=True).mean() >= 0.3
        numeric_ratio = pd.to_numeric(sample, errors="coerce").notna().mean()

        if not has_datetime_separators and numeric_ratio < 0.7:
            continue

        parsed = parse_timestamp_column(sample)
        by_values = parsed.notna().mean() >= 0.7

        if by_values:
            candidates.append(col)

    return candidates


def get_price_candidate_columns(df):
    price_tokens = (
        "close",
        "open",
        "high",
        "low",
        "price",
        "last",
    )

    candidates = []
    for col in df.columns:
        normalized = normalize_name(col)
        numeric_ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()
        if numeric_ratio < 0.5:
            continue

        is_price_name = any(token in normalized for token in price_tokens)
        is_non_price_metric = (
            "volume" in normalized
            or "count" in normalized
            or "trade" in normalized
            or "qty" in normalized
        )

        if is_price_name and not is_non_price_metric:
            candidates.append(col)

    return candidates


def get_default_price_column(price_columns):
    preferred_price_order = ["close", "open", "high", "low"]
    normalized_to_original = {normalize_name(col): col for col in price_columns}

    for candidate in preferred_price_order:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]

    return price_columns[0] if price_columns else None


def parse_timestamp_column(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.7:
        best = None
        best_valid_ratio = 0.0
        for unit in ("s", "ms", "us", "ns"):
            parsed = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
            valid = parsed.notna()
            if valid.any():
                years = parsed[valid].dt.year
                plausible = years.between(2009, 2100).mean()
                if plausible > best_valid_ratio:
                    best_valid_ratio = plausible
                    best = parsed
        if best is not None and best_valid_ratio >= 0.7:
            return best.dt.tz_localize(None)

    parsed_fallback = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed_fallback.dt.tz_localize(None)


@st.cache_data(show_spinner=False)
def load_csv_from_path(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_csv_from_bytes(content):
    return pd.read_csv(BytesIO(content))


@st.cache_data(show_spinner="Loading default BTC dataset...")
def load_default_dataset():
    """Load local default CSV, or fall back to the public Kaggle dataset."""
    if DEFAULT_DATASET_PATH.exists():
        return load_csv_from_path(str(DEFAULT_DATASET_PATH)), DEFAULT_DATASET_PATH.name, "local"

    try:
        import kagglehub
    except ImportError as exc:
        raise FileNotFoundError(
            "Default local dataset is missing and Kaggle fallback is unavailable. "
            "Switch to Upload CSV or install kagglehub."
        ) from exc

    try:
        dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_REF))
        preferred_path = dataset_dir / KAGGLE_DATASET_FILE

        if preferred_path.exists():
            csv_path = preferred_path
        else:
            matches = list(dataset_dir.rglob(KAGGLE_DATASET_FILE))
            if not matches:
                raise FileNotFoundError(
                    f"Could not locate '{KAGGLE_DATASET_FILE}' in downloaded Kaggle dataset."
                )
            csv_path = matches[0]

        source_name = f"Kaggle ({KAGGLE_DATASET_REF})"
        return load_csv_from_path(str(csv_path)), source_name, "kaggle"
    except Exception as exc:
        raise FileNotFoundError(
            "Default local dataset is missing and automatic Kaggle download failed. "
            "Switch to Upload CSV or add datasets/btcusd_1-min_data.csv."
        ) from exc


@st.cache_data(show_spinner="Preparing daily series…")
def prepare_daily_series(df, timestamp_col, price_col):
    temp = df[[timestamp_col, price_col]].copy()
    temp.columns = ["date", "price"]
    temp["date"] = parse_timestamp_column(temp["date"])
    temp["price"] = pd.to_numeric(temp["price"], errors="coerce")
    temp = temp.dropna(subset=["date", "price"])

    if temp.empty:
        raise ValueError(
            "After parsing, no valid (date, price) pairs remain. "
            "Please check your column selections."
        )

    was_sorted = temp["date"].is_monotonic_increasing
    temp = temp.sort_values("date")
    duplicate_timestamps_removed = int(temp.duplicated(subset="date", keep="last").sum())
    temp = temp.drop_duplicates(subset="date", keep="last")

    temp = temp.set_index("date")
    temp = temp.asfreq("D")
    missing_days = int(temp["price"].isna().sum())
    temp["price"] = temp["price"].ffill()

    temp_series = temp["price"]
    prepared = temp_series.reset_index()
    prepared.columns = ["ds", "y"]

    diagnostics = {
        "was_sorted": was_sorted,
        "duplicate_timestamps_removed": duplicate_timestamps_removed,
        "missing_days": missing_days,
        "rows": len(prepared),
        "start": prepared["ds"].min(),
        "end": prepared["ds"].max(),
    }
    return prepared, diagnostics


def downsample_for_plot(df, max_points=2500):
    if len(df) <= max_points:
        return df.reset_index(drop=True)

    step = int(np.ceil(len(df) / max_points))
    sampled = df.iloc[::step].copy()
    last_row = df.iloc[[-1]]

    sampled_last_row = sampled.iloc[[-1]].reset_index(drop=True)
    original_last_row = last_row.reset_index(drop=True)
    if not sampled_last_row.equals(original_last_row):
        sampled = pd.concat([sampled, last_row])

    return sampled.reset_index(drop=True)


def select_training_frame(prepared_df, training_window_days):
    """Select the most recent slice used for model training/backtesting."""
    if training_window_days is None or training_window_days >= len(prepared_df):
        return prepared_df.copy()
    return prepared_df.tail(training_window_days).copy()


def split_train_test(series, horizon_days):
    n = len(series)
    if n < 80:
        raise ValueError(
            "Not enough daily records. Please provide a longer BTC history (>= 80 days)."
        )
    test_size = max(horizon_days, int(n * 0.2))
    test_size = min(test_size, n - 30)
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test


def ci_to_z_score(confidence_percent):
    return NormalDist().inv_cdf(0.5 + confidence_percent / 200.0)


def calculate_mape(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    non_zero_mask = y_true_arr != 0

    if not np.any(non_zero_mask):
        return np.nan

    mape = np.mean(np.abs((y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask]))
    return float(mape * 100)


def extract_confidence_bounds(conf_int_df, target_index):
    lower_col = None
    upper_col = None

    for col in conf_int_df.columns:
        col_name = str(col).lower()
        if lower_col is None and col_name.startswith("lower"):
            lower_col = col
        if upper_col is None and col_name.startswith("upper"):
            upper_col = col

    if lower_col is None or upper_col is None:
        if conf_int_df.shape[1] >= 2:
            lower_col = conf_int_df.columns[0]
            upper_col = conf_int_df.columns[1]
        else:
            raise ValueError("Could not determine lower/upper confidence interval columns.")

    lower_series = pd.Series(conf_int_df[lower_col].values, index=target_index)
    upper_series = pd.Series(conf_int_df[upper_col].values, index=target_index)
    return lower_series, upper_series


def run_arima_forecast(series, horizon_days, confidence_percent):
    ARIMA, auto_arima = _import_arima_stack()

    train, test = split_train_test(series, horizon_days)
    alpha = 1 - (confidence_percent / 100.0)

    auto_model = auto_arima(
        train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
        information_criterion="aic",
        max_p=3,
        max_q=3,
        max_d=2,
    )
    best_order = auto_model.order

    backtest_model = ARIMA(train, order=best_order).fit()
    backtest_forecast = backtest_model.get_forecast(steps=len(test))
    backtest_pred = pd.Series(backtest_forecast.predicted_mean.values, index=test.index)
    backtest_ci = backtest_forecast.conf_int(alpha=alpha)
    backtest_lower, backtest_upper = extract_confidence_bounds(backtest_ci, test.index)

    final_model = ARIMA(series, order=best_order).fit()
    future_forecast = final_model.get_forecast(steps=horizon_days)
    future_idx = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )
    future_pred = pd.Series(future_forecast.predicted_mean.values, index=future_idx)
    future_ci = future_forecast.conf_int(alpha=alpha)
    future_lower, future_upper = extract_confidence_bounds(future_ci, future_idx)

    return {
        "test": test,
        "backtest_pred": backtest_pred,
        "backtest_lower": backtest_lower,
        "backtest_upper": backtest_upper,
        "arima_order": best_order,
        "future_pred": future_pred,
        "future_lower": future_lower,
        "future_upper": future_upper,
    }


def run_prophet_forecast(series, horizon_days, confidence_percent):
    Prophet = _import_prophet()

    train, test = split_train_test(series, horizon_days)
    interval_width = confidence_percent / 100.0

    train_df = train.reset_index()
    train_df.columns = ["ds", "y"]

    backtest_model = Prophet(interval_width=interval_width, daily_seasonality="auto")
    backtest_model.fit(train_df)

    test_df = pd.DataFrame({"ds": test.index})
    backtest_forecast_df = backtest_model.predict(test_df)

    backtest_pred = pd.Series(backtest_forecast_df["yhat"].values, index=test.index)
    backtest_lower = pd.Series(backtest_forecast_df["yhat_lower"].values, index=test.index)
    backtest_upper = pd.Series(backtest_forecast_df["yhat_upper"].values, index=test.index)

    final_training_df = series.reset_index()
    final_training_df.columns = ["ds", "y"]

    final_model = Prophet(interval_width=interval_width, daily_seasonality="auto")
    final_model.fit(final_training_df)

    future_df = final_model.make_future_dataframe(periods=horizon_days, freq="D")
    future_forecast_df = final_model.predict(future_df)

    future_rows = future_forecast_df[future_forecast_df["ds"] > series.index[-1]].copy()
    future_idx = pd.DatetimeIndex(future_rows["ds"].values)

    future_pred = pd.Series(future_rows["yhat"].values, index=future_idx)
    future_lower = pd.Series(future_rows["yhat_lower"].values, index=future_idx)
    future_upper = pd.Series(future_rows["yhat_upper"].values, index=future_idx)

    return {
        "test": test,
        "backtest_pred": backtest_pred,
        "backtest_lower": backtest_lower,
        "backtest_upper": backtest_upper,
        "future_pred": future_pred,
        "future_lower": future_lower,
        "future_upper": future_upper,
    }


def create_lstm_sequences(data, lookback=14):
    """Create sliding-window sequences for LSTM input (vectorised)."""
    n = len(data) - lookback
    idx = np.arange(lookback)[None, :] + np.arange(n)[:, None]
    X = data[idx, 0]
    y = data[lookback:, 0]
    return X, y


def build_lstm_model(Sequential, LSTM, Dense, Dropout, lookback):
    """Build a compact LSTM model."""
    model = Sequential([
        LSTM(48, return_sequences=False, input_shape=(lookback, 1)),
        Dropout(0.15),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def run_lstm_forecast(series, horizon_days, confidence_percent):
    """LSTM-based forecast with backtest evaluation and future projection."""
    if not is_lstm_available():
        raise ImportError(
            "LSTM requires TensorFlow, which is not installed for this Python runtime. "
            "Use ARIMA/Prophet/Holt-Winters or run on Python 3.11/3.12 for LSTM."
        )

    Sequential, LSTM_Layer, Dense, Dropout, EarlyStopping, MinMaxScaler = _import_lstm_stack()

    train, test = split_train_test(series, horizon_days)
    lookback = 14

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))

    X_train, y_train = create_lstm_sequences(train_scaled, lookback)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    backtest_model = build_lstm_model(Sequential, LSTM_Layer, Dense, Dropout, lookback)
    early_stop = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    backtest_model.fit(
        X_train, y_train,
        epochs=30, batch_size=64,
        callbacks=[early_stop], verbose=0,
    )

    combined_backtest = pd.concat([train.iloc[-lookback:], test])
    combined_scaled = scaler.transform(combined_backtest.values.reshape(-1, 1))
    X_test, _ = create_lstm_sequences(combined_scaled, lookback)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    backtest_pred_scaled = backtest_model.predict(X_test, verbose=0)
    backtest_pred_values = scaler.inverse_transform(backtest_pred_scaled).flatten()
    backtest_pred = pd.Series(backtest_pred_values, index=test.index[:len(backtest_pred_values)])

    train_pred_scaled = backtest_model.predict(X_train, verbose=0)
    train_pred_values = scaler.inverse_transform(train_pred_scaled).flatten()
    train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    residuals = train_actual - train_pred_values
    sigma = float(np.std(residuals, ddof=1))
    z = ci_to_z_score(confidence_percent)

    backtest_lower = backtest_pred - z * sigma
    backtest_upper = backtest_pred + z * sigma

    full_scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X_full, y_full = create_lstm_sequences(full_scaled, lookback)
    X_full = X_full.reshape((X_full.shape[0], X_full.shape[1], 1))

    final_model = build_lstm_model(Sequential, LSTM_Layer, Dense, Dropout, lookback)
    early_stop2 = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
    final_model.fit(
        X_full, y_full,
        epochs=30, batch_size=64,
        callbacks=[early_stop2], verbose=0,
    )

    last_window = full_scaled[-lookback:].flatten().tolist()
    future_idx = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )
    future_values_scaled = []
    for _ in range(horizon_days):
        x_input = np.array(last_window[-lookback:]).reshape(1, lookback, 1)
        next_val = final_model.predict(x_input, verbose=0)[0, 0]
        future_values_scaled.append(next_val)
        last_window.append(next_val)

    future_values = scaler.inverse_transform(
        np.array(future_values_scaled).reshape(-1, 1)
    ).flatten()
    future_pred = pd.Series(future_values, index=future_idx)
    future_lower = future_pred - z * sigma
    future_upper = future_pred + z * sigma

    return {
        "test": test,
        "backtest_pred": backtest_pred,
        "backtest_lower": backtest_lower,
        "backtest_upper": backtest_upper,
        "future_pred": future_pred,
        "future_lower": future_lower,
        "future_upper": future_upper,
    }


def run_holt_winters(series, horizon_days, confidence_percent):
    """Holt-Winters Exponential Smoothing forecast."""
    ExponentialSmoothing = _import_holt_winters()

    train, test = split_train_test(series, horizon_days)
    alpha = 1 - (confidence_percent / 100.0)

    seasonal_periods = 7
    use_seasonal = len(train) >= 2 * seasonal_periods

    if use_seasonal:
        backtest_model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
        ).fit(optimized=True)
    else:
        backtest_model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal=None,
        ).fit(optimized=True)

    backtest_pred = backtest_model.forecast(steps=len(test))
    backtest_pred.index = test.index

    train_fitted = backtest_model.fittedvalues
    residuals = train.loc[train_fitted.index] - train_fitted
    sigma = float(residuals.std(ddof=1))
    z = ci_to_z_score(confidence_percent)

    backtest_lower = backtest_pred - z * sigma
    backtest_upper = backtest_pred + z * sigma

    if use_seasonal:
        final_model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
        ).fit(optimized=True)
    else:
        final_model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=None,
        ).fit(optimized=True)

    future_idx = pd.date_range(
        start=series.index[-1] + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )
    future_pred = final_model.forecast(steps=horizon_days)
    future_pred.index = future_idx

    future_lower = future_pred - z * sigma
    future_upper = future_pred + z * sigma

    return {
        "test": test,
        "backtest_pred": backtest_pred,
        "backtest_lower": backtest_lower,
        "backtest_upper": backtest_upper,
        "future_pred": future_pred,
        "future_lower": future_lower,
        "future_upper": future_upper,
    }


@st.cache_data(show_spinner=False)
def generate_forecast_results(training_df, model_choice, horizon_days, confidence_pct):
    series = training_df.set_index("ds")["y"]

    if model_choice == "ARIMA":
        return run_arima_forecast(series, horizon_days, confidence_pct)
    if model_choice == "Prophet":
        return run_prophet_forecast(series, horizon_days, confidence_pct)
    if model_choice == "Holt-Winters":
        return run_holt_winters(series, horizon_days, confidence_pct)
    return run_lstm_forecast(series, horizon_days, confidence_pct)
