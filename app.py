"""
app.py – Bitcoin Price Forecasting Portal
Redesigned UI/UX layer.  All model logic lives in forecasting.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

from forecasting import (
    DEFAULT_DATASET_PATH,
    calculate_mape,
    downsample_for_plot,
    find_timestamp_column,
    generate_forecast_results,
    get_default_price_column,
    get_price_candidate_columns,
    get_timestamp_candidate_columns,
    load_csv_from_bytes,
    load_csv_from_path,
    prepare_daily_series,
    select_training_frame,
)

# ── colour palette ──────────────────────────────────────────────────────────
BTC_ORANGE   = "#F7931A"
BTC_GOLD     = "#FFB347"
ACCENT_TEAL  = "#00D4AA"
ACCENT_RED   = "#FF4B6E"
SURFACE_DARK = "#161B22"
TEXT_DIM     = "#8B949E"
PLOT_BG      = "#0D1117"
GRID_CLR     = "#21262D"

# ── model metadata (icons + descriptions for the selector) ──────────────────
MODEL_INFO = {
    "ARIMA": {
        "icon": "📈",
        "desc": "Auto-tuned ARIMA – captures autocorrelation & short-term momentum.",
    },
    "Prophet": {
        "icon": "🔮",
        "desc": "Facebook Prophet – robust to missing data with trend & seasonality.",
    },
    "LSTM (Deep Learning)": {
        "icon": "🧠",
        "desc": "Recurrent neural network that learns complex non-linear patterns.",
    },
    "Holt-Winters": {
        "icon": "❄️",
        "desc": "Exponential smoothing with trend & weekly seasonal components.",
    },
}

# ── custom CSS ──────────────────────────────────────────────────────────────
_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global font */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero header */
.hero-title {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #F7931A 0%, #FFB347 50%, #F7931A 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0;
}
.hero-sub { color: #8B949E; font-size: 1.05rem; margin-top: 4px; }

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #161B22 0%, #1A1D23 100%);
    border: 1px solid #21262D; border-radius: 14px;
    padding: 20px 18px; text-align: center;
    transition: transform .18s, border-color .18s;
}
.metric-card:hover { transform: translateY(-3px); border-color: #F7931A55; }
.metric-card .label { font-size: .78rem; color: #8B949E; text-transform: uppercase;
    letter-spacing: .06em; margin-bottom: 6px; }
.metric-card .value { font-size: 1.55rem; font-weight: 700; color: #FAFAFA;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 100%; }
.metric-card .value.orange { color: #F7931A; }
.metric-card .value.teal   { color: #00D4AA; }
.metric-card .value.red    { color: #FF4B6E; }
.metric-card .value.small  { font-size: 1rem; }

/* Section headers */
.section-hdr {
    font-size: 1.25rem; font-weight: 700; color: #FAFAFA;
    border-left: 4px solid #F7931A; padding-left: 12px;
    margin: 30px 0 14px 0;
}

/* Sidebar polish */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%) !important;
    border-right: 1px solid #21262D !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 { color: #F7931A !important; }

/* Divider */
.orange-rule { border: none; border-top: 2px solid #F7931A44; margin: 18px 0; }

/* Data-source tabs */
.data-tab { font-weight: 600; }

/* Footer */
.app-footer {
    text-align: center; color: #484F58; font-size: .78rem;
    padding: 30px 0 10px 0; border-top: 1px solid #21262D; margin-top: 40px;
}
</style>
"""


# ── helper: html metric card ───────────────────────────────────────────────
def _metric_html(label: str, value: str, color_class: str = "") -> str:
    return (
        f'<div class="metric-card">'
        f'  <div class="label">{label}</div>'
        f'  <div class="value {color_class}">{value}</div>'
        f'</div>'
    )


# ── Plotly layout defaults (dark theme) ────────────────────────────────────
_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family="Inter, sans-serif", color="#C9D1D9"),
    xaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR),
    yaxis=dict(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR, tickprefix="$", tickformat=",.0f"),
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        bgcolor="rgba(0,0,0,0)", font=dict(size=12),
    ),
    margin=dict(l=20, r=20, t=60, b=30),
)


def _build_history_figure(prepared_df: pd.DataFrame) -> go.Figure:
    """Interactive historical price chart with range selector."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prepared_df["ds"], y=prepared_df["y"],
        mode="lines", name="BTC Daily Close",
        line=dict(color=BTC_ORANGE, width=2),
        fill="tozeroy", fillcolor="rgba(247,147,26,0.07)",
    ))
    fig.update_layout(**_PLOTLY_LAYOUT, title=dict(text="Historical BTC Price", font=dict(size=18)))
    fig.update_layout(
        xaxis=dict(
            gridcolor=GRID_CLR, zerolinecolor=GRID_CLR,
            rangeselector=dict(
                buttons=[
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#161B22", activecolor=BTC_ORANGE,
                font=dict(color="#FAFAFA"),
            ),
            rangeslider=dict(visible=True, bgcolor="#0D1117"),
            type="date",
        ),
    )
    return fig


def _build_forecast_figure(
    history_df: pd.DataFrame,
    future_pred: pd.Series,
    future_lower: pd.Series,
    future_upper: pd.Series,
) -> go.Figure:
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=history_df["ds"], y=history_df["y"],
        mode="lines", name="Historical Price",
        line=dict(color=BTC_ORANGE, width=2),
    ))

    # CI band
    fig.add_trace(go.Scatter(
        x=future_pred.index, y=future_upper.values,
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=future_pred.index, y=future_lower.values,
        mode="lines", fill="tonexty",
        fillcolor="rgba(0,212,170,0.12)", line=dict(width=0),
        name="Confidence Band", hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_pred.index, y=future_pred.values,
        mode="lines", name="Forecast",
        line=dict(color=ACCENT_TEAL, width=2.8, dash="dot"),
    ))

    # Anchor marker
    fig.add_trace(go.Scatter(
        x=[history_df["ds"].iloc[-1]],
        y=[float(history_df["y"].iloc[-1])],
        mode="markers", name="Forecast Start",
        marker=dict(size=11, color=ACCENT_TEAL, symbol="diamond",
                     line=dict(width=2, color="#0D1117")),
    ))

    # Vertical separator
    forecast_start = future_pred.index[0]
    fig.add_shape(
        type="line", x0=forecast_start, x1=forecast_start,
        y0=0, y1=1, xref="x", yref="paper",
        line=dict(color=ACCENT_TEAL, width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=forecast_start, y=1.0, xref="x", yref="paper",
        text="  ▶ Forecast", showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(color=ACCENT_TEAL, size=12, family="Inter"),
    )

    # Auto-zoom: show recent context (3× forecast horizon) + full forecast
    forecast_days = len(future_pred)
    context_days = max(forecast_days * 3, 90)  # at least 90 days of history
    zoom_start = forecast_start - pd.Timedelta(days=context_days)
    zoom_end = future_pred.index[-1] + pd.Timedelta(days=5)  # small padding

    fig.update_layout(**_PLOTLY_LAYOUT, title=dict(text="Bitcoin Price Forecast", font=dict(size=18)))
    fig.update_layout(
        xaxis=dict(
            gridcolor=GRID_CLR, zerolinecolor=GRID_CLR,
            range=[zoom_start, zoom_end],
            rangeselector=dict(
                buttons=[
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#161B22", activecolor=BTC_ORANGE,
                font=dict(color="#FAFAFA"),
            ),
            type="date",
        ),
    )
    return fig


def _build_backtest_figure(
    test: pd.Series,
    backtest_pred: pd.Series,
    backtest_lower: pd.Series,
    backtest_upper: pd.Series,
) -> go.Figure:
    """Backtest: actual vs predicted on the hold-out set."""
    common = test.index.intersection(backtest_pred.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=common, y=test.loc[common].values,
        mode="lines", name="Actual",
        line=dict(color=BTC_ORANGE, width=2),
    ))
    # CI
    fig.add_trace(go.Scatter(
        x=common, y=backtest_upper.loc[common].values,
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=common, y=backtest_lower.loc[common].values,
        mode="lines", fill="tonexty",
        fillcolor="rgba(255,75,110,0.10)", line=dict(width=0),
        name="Confidence Band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=common, y=backtest_pred.loc[common].values,
        mode="lines", name="Predicted",
        line=dict(color=ACCENT_RED, width=2, dash="dot"),
    ))
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="Back-test: Actual vs Predicted", font=dict(size=16)),
    )
    return fig


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    st.set_page_config(
        page_title="₿ BTC Forecast",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown('<p class="hero-title">₿ Bitcoin Forecasting Portal</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">Upload a BTC CSV · pick a model · generate forecasts with confidence bands</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="orange-rule">', unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        # Data source
        st.markdown("### 📂 Data Source")
        data_source = st.radio(
            "Source", ["Default dataset", "Upload CSV"],
            horizontal=True, label_visibility="collapsed",
        )

    # ── Load data ───────────────────────────────────────────────────────────
    if data_source == "Upload CSV":
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload BTC CSV", type=["csv"])
        if uploaded_file is None:
            st.info(
                "📤 **Upload a CSV file** to begin.\n\n"
                "Compatible datasets:\n"
                "- [Kaggle BTC Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)\n"
                "- Any CSV with a date/timestamp column + a numeric price column."
            )
            st.stop()
        raw_df = load_csv_from_bytes(uploaded_file.getvalue())
        source_name = uploaded_file.name
    else:
        if not DEFAULT_DATASET_PATH.exists():
            st.error(
                f"Default local dataset not found: `{DEFAULT_DATASET_PATH}`. "
                "Switch to **Upload CSV** or add the file."
            )
            st.stop()
        raw_df = load_csv_from_path(str(DEFAULT_DATASET_PATH))
        source_name = DEFAULT_DATASET_PATH.name

    if raw_df.empty:
        st.error("The uploaded CSV is empty.")
        st.stop()
    if len(raw_df.columns) < 2:
        st.error("The CSV needs at least a timestamp column and a price column.")
        st.stop()

    # ── Column detection ────────────────────────────────────────────────────
    timestamp_columns = get_timestamp_candidate_columns(raw_df)
    if not timestamp_columns:
        st.error("No timestamp-like columns detected.")
        st.stop()

    price_columns = get_price_candidate_columns(raw_df)
    if not price_columns:
        st.error("No price-like numeric columns detected.")
        st.stop()

    timestamp_guess = find_timestamp_column(timestamp_columns)
    ts_idx = timestamp_columns.index(timestamp_guess) if timestamp_guess in timestamp_columns else 0
    default_price_col = get_default_price_column(price_columns)

    # ── Sidebar controls ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🗂️ Columns")
        timestamp_col = st.selectbox("Timestamp column", timestamp_columns, index=ts_idx)
        price_col = st.selectbox("Price column", price_columns,
                                  index=price_columns.index(default_price_col))

        st.markdown("---")
        st.markdown("### 🤖 Model")
        model_choice = st.selectbox(
            "Forecasting model",
            list(MODEL_INFO.keys()),
            format_func=lambda m: f"{MODEL_INFO[m]['icon']}  {m}",
        )
        st.caption(MODEL_INFO[model_choice]["desc"])

        st.markdown("---")
        st.markdown("### 📐 Parameters")
        horizon_days = st.slider("Forecast horizon (days)", 7, 365, 30)
        confidence_pct = st.slider("Confidence interval (%)", 80, 99, 95)

    # ── Prepare data ────────────────────────────────────────────────────────
    try:
        prepared_df, diagnostics = prepare_daily_series(raw_df, timestamp_col, price_col)
    except ValueError as exc:
        st.error(f"Data preparation failed: {exc}")
        st.stop()

    # ── Training window ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🪟 Training Window")
        tw_choice = st.selectbox(
            "Window",
            ["Last 365 days ★", "Last 180 days", "Last 730 days", "All history", "Custom"],
            label_visibility="collapsed",
        )
        if tw_choice == "Last 180 days":
            tw_days = 180
        elif tw_choice.startswith("Last 365"):
            tw_days = 365
        elif tw_choice == "Last 730 days":
            tw_days = 730
        elif tw_choice == "Custom":
            max_cd = max(80, len(prepared_df))
            tw_days = int(st.number_input("Custom window (days)", 80, max_cd,
                                           min(365, max_cd), step=10))
        else:
            tw_days = None

    training_df = select_training_frame(prepared_df, tw_days)
    training_start = training_df["ds"].min()
    training_end = training_df["ds"].max()

    max_stable_horizon = max(7, len(training_df) - 30)
    effective_horizon = min(horizon_days, max_stable_horizon)
    if effective_horizon != horizon_days:
        st.warning(
            f"Horizon reduced from {horizon_days} → {effective_horizon} days "
            f"(training window has {len(training_df)} rows)."
        )

    # ── Data-quality warnings ───────────────────────────────────────────────
    if not diagnostics["was_sorted"]:
        st.warning("⚠️ Rows were not chronological – they've been sorted automatically.")
    if diagnostics["duplicate_timestamps_removed"] > 0:
        st.warning(
            f"⚠️ {diagnostics['duplicate_timestamps_removed']} duplicate timestamps removed "
            f"(kept last value for *{price_col}*)."
        )

    # ── Dashboard metrics row ───────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📊 Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_metric_html("Source File", source_name, "small"), unsafe_allow_html=True)
    with c2:
        st.markdown(_metric_html("Daily Records", f"{diagnostics['rows']:,}", "orange"),
                     unsafe_allow_html=True)
    with c3:
        st.markdown(_metric_html("Gap-Filled Days", str(diagnostics["missing_days"]), "teal"),
                     unsafe_allow_html=True)
    with c4:
        st.markdown(_metric_html("Training Rows", f"{len(training_df):,}", "orange"),
                     unsafe_allow_html=True)

    st.caption(
        f"📅 Full range **{diagnostics['start'].date()}** → **{diagnostics['end'].date()}** · "
        f"Training on **{training_start.date()}** → **{training_end.date()}** · "
        f"Price column: `{price_col}`"
    )

    # ── Historical chart ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📉 Historical Price</div>', unsafe_allow_html=True)
    st.plotly_chart(
        _build_history_figure(downsample_for_plot(prepared_df)),
        use_container_width=True,
    )

    # ── Forecast button ─────────────────────────────────────────────────────
    st.markdown('<hr class="orange-rule">', unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        run_forecast = st.button(
            f"🚀  Generate {model_choice} Forecast  ({effective_horizon}d)",
            type="primary",
            use_container_width=True,
        )

    if not run_forecast:
        st.stop()

    # ── Run forecast ────────────────────────────────────────────────────────
    with st.spinner(f"Training **{model_choice}** and projecting {effective_horizon} days…"):
        try:
            results = generate_forecast_results(
                training_df, model_choice, effective_horizon, confidence_pct,
            )
        except ImportError as exc:
            st.error(f"Missing dependency: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Forecast failed: {exc}")
            st.stop()

    # ── Backtest metrics ────────────────────────────────────────────────────
    common_idx = results["test"].index.intersection(results["backtest_pred"].index)
    y_true = results["test"].loc[common_idx]
    y_pred = results["backtest_pred"].loc[common_idx]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = calculate_mape(y_true, y_pred)

    st.markdown('<div class="section-hdr">🎯 Back-test Accuracy</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(_metric_html("MAE", f"${mae:,.2f}", "teal"), unsafe_allow_html=True)
    with m2:
        st.markdown(_metric_html("RMSE", f"${rmse:,.2f}", "orange"), unsafe_allow_html=True)
    with m3:
        mape_str = f"{mape:.2f}%" if not np.isnan(mape) else "N/A"
        clr = "teal" if (not np.isnan(mape) and mape < 5) else ("orange" if not np.isnan(mape) and mape < 15 else "red")
        st.markdown(_metric_html("MAPE", mape_str, clr), unsafe_allow_html=True)

    if model_choice == "ARIMA" and "arima_order" in results:
        st.caption(f"🔧 Selected ARIMA order *(p, d, q)*: **{results['arima_order']}**")

    # ── Backtest chart ──────────────────────────────────────────────────────
    st.plotly_chart(
        _build_backtest_figure(
            results["test"], results["backtest_pred"],
            results["backtest_lower"], results["backtest_upper"],
        ),
        use_container_width=True,
    )

    # ── Forecast chart ──────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🔭 Future Forecast</div>', unsafe_allow_html=True)

    # Summary cards for forecast
    last_price = float(prepared_df["y"].iloc[-1])
    end_price  = float(results["future_pred"].iloc[-1])
    pct_change = ((end_price - last_price) / last_price) * 100
    direction  = "🟢" if pct_change >= 0 else "🔴"

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(_metric_html("Current Price", f"${last_price:,.2f}", "orange"),
                     unsafe_allow_html=True)
    with s2:
        st.markdown(_metric_html(f"Day-{effective_horizon} Forecast", f"${end_price:,.2f}", "teal"),
                     unsafe_allow_html=True)
    with s3:
        st.markdown(
            _metric_html(f"{direction} Expected Change", f"{pct_change:+.2f}%",
                          "teal" if pct_change >= 0 else "red"),
            unsafe_allow_html=True,
        )

    st.plotly_chart(
        _build_forecast_figure(
            history_df=downsample_for_plot(prepared_df, max_points=2500),
            future_pred=results["future_pred"],
            future_lower=results["future_lower"],
            future_upper=results["future_upper"],
        ),
        use_container_width=True,
    )

    # ── Forecast table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📋 Forecast Data</div>', unsafe_allow_html=True)

    forecast_table = pd.DataFrame({
        "Date": results["future_pred"].index.strftime("%Y-%m-%d"),
        "Predicted (USD)": results["future_pred"].values.round(2),
        f"Lower {confidence_pct}%": results["future_lower"].values.round(2),
        f"Upper {confidence_pct}%": results["future_upper"].values.round(2),
    })

    st.dataframe(
        forecast_table.style.format({
            "Predicted (USD)": "${:,.2f}",
            f"Lower {confidence_pct}%": "${:,.2f}",
            f"Upper {confidence_pct}%": "${:,.2f}",
        }),
        use_container_width=True,
        height=min(400, 35 * len(forecast_table) + 38),
    )

    # ── CSV download ────────────────────────────────────────────────────────
    csv_data = forecast_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download Forecast CSV",
        data=csv_data,
        file_name=f"btc_forecast_{model_choice.lower().replace(' ', '_')}_{effective_horizon}d.csv",
        mime="text/csv",
    )

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="app-footer">'
        'Built with Streamlit · Plotly · Pandas · scikit-learn<br>'
        '₿ Bitcoin Forecasting Portal'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
