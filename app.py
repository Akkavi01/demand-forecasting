# app.py — Streamlit Forecasting (Prophet & SARIMAX)
# --------------------------------------------------
# • Upload CSV (date, demand; optional: price, promo)
# • Split: last N periods as test
# • Models: Prophet or SARIMAX
# • Metrics on test: MAE, RMSE, MAPE
# • Interactive Plotly charts
# • Download forecast CSV


import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Prophet (optional)
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    try:
        from fbprophet import Prophet  # legacy fallback
        HAVE_PROPHET = True
    except Exception:
        HAVE_PROPHET = False

import requests


from groq import Groq
import streamlit as st

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Groq API key missing")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)
#----------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Forecasting (Prophet & SARIMAX)", layout="wide")
st.title("📈 Demand Forecasting")
# st.caption("Upload your time series, split into train/test, compare models, and download forecasts.")

# -------------------------------
# Helpers
# -------------------------------
def infer_freq(idx: pd.DatetimeIndex) -> str:
    f = pd.infer_freq(idx)
    return f if f else "M"

@st.cache_data(show_spinner=False)
def read_csv(upload, date_col: str, y_col: str) -> pd.DataFrame:
    df = pd.read_csv(upload)
    if date_col not in df.columns or y_col not in df.columns:
        raise ValueError("Date or demand column missing in file.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[y_col])
    return df

def rmse_compat(y_true, y_pred) -> float:
    # Works for any sklearn version
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape_compat(y_true, y_pred) -> float:
    arr = (np.abs((y_true - y_pred) / y_true)).replace([np.inf, -np.inf], np.nan)
    return float(np.nanmean(arr) * 100.0)




@st.cache_data(show_spinner=False)
def explain_with_llm(
    model_choice,
    freq,
    train,
    test,
    forecast,
    mae,
    rmse,
    mape,
    exog_cols
):
    prompt = f"""
You are a senior demand forecasting consultant.

Model used: {model_choice}
Data frequency: {freq}

Training period:
{train.index.min().date()} to {train.index.max().date()}

Test period:
{test.index.min().date()} to {test.index.max().date()}

Forecast accuracy:
- MAE: {mae:.2f}
- RMSE: {rmse:.2f}
- MAPE: {mape:.2f}%

Exogenous variables:
{exog_cols if exog_cols else "None"}

Recent forecast values:
{forecast.tail(6).to_string()}

Explain clearly:
1. Trend and seasonality
2. Forecast reliability
3. Risks or anomalies
4. Business recommendations

Use simple business language.
"""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a demand forecasting expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"❌ LLM Error: {str(e)}"




#----------------------------------------------------------------------------------------------------------







# -------------------------------
# Sidebar controls (ONE uploader with unique key)
# -------------------------------
with st.sidebar:
    st.header("1) Data")
    upload = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="csv_uploader_main",  # unique key
        help="Columns expected: date, demand (optional: price, promo)"
    )
    date_col = st.text_input("Date column", value="date")
    y_col = st.text_input("Demand column", value="demand")

    st.header("2) Split")
    holdout = st.number_input("Test size (last N periods)", min_value=1, max_value=120, value=12)

    st.header("3) Model")
    model_choices = ["SARIMAX"] + (["Prophet"] if HAVE_PROPHET else [])
    model_choice = st.selectbox("Choose model", model_choices, index=0)

    if model_choice == "SARIMAX":
        st.caption("SARIMAX: ARIMA(p,d,q) × (P,D,Q)[m]")
        p = st.number_input("p", 0, 5, 1)
        d = st.number_input("d", 0, 2, 1)
        q = st.number_input("q", 0, 5, 1)
        P = st.number_input("P", 0, 5, 1)
        D = st.number_input("D", 0, 2, 1)
        Q = st.number_input("Q", 0, 5, 1)
        m = st.number_input("Seasonal period (m)", 1, 366, 12)
    else:  # Prophet
        seasonality_mode = st.selectbox("Seasonality mode", ["multiplicative", "additive"], index=0)
        cps = st.slider("Changepoint prior scale", 0.01, 1.0, 0.5, 0.01)

    st.header("4) Forecast")
    future_h = st.number_input("Future horizon (after test)", 0, 60, 0,
                               help="Extra periods to forecast beyond the test window")

if upload is None:
    st.info("⬅️ Upload a CSV to begin.")
    st.stop()

# -------------------------------
# Load & prepare
# -------------------------------
try:
    df_raw = read_csv(upload, date_col, y_col)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

all_cols = df_raw.columns.tolist()
exog_candidates = [c for c in ["price", "promo"] if c in all_cols]

freq = infer_freq(df_raw.index)

# Aggregate to regular frequency (sum demand, mean exog)
agg = {y_col: "sum"}
for c in exog_candidates:
    agg[c] = "mean"
df = df_raw.resample(freq).agg(agg)

# Ensure enough data
if len(df) < holdout + 12:
    st.warning("Not enough history for the requested holdout. Reduce test size or provide more data.")
    st.stop()

# Split train/test
train = df.iloc[:-holdout].copy()
test = df.iloc[-holdout:].copy()

st.subheader("Data preview")
col1, col2 = st.columns(2)
with col1:
    st.write("Train tail")
    st.write(train.tail())
with col2:
    st.write("Test head")
    st.write(test.head())

# -------------------------------
# Fit & forecast helpers
# -------------------------------
def fit_predict_sarimax(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        extra_h: int,
                        order, seas_order,
                        freq: str,
                        exog_cols) -> pd.DataFrame:
    """
    Fit SARIMAX on train and forecast len(test)+extra_h steps.
    Robust exog handling without manual index tweaking.
    """
    y_train = train_df[y_col]
    y_test = test_df[y_col]

    # Build exog blocks
    exog_train = train_df[exog_cols] if exog_cols else None
    exog_test = test_df[exog_cols] if exog_cols else None

    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seas_order,
        exog=exog_train,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)

    steps = len(y_test) + int(extra_h)

    # Build exog for forecast steps (test + future), repeating last known row
    exog_steps = None
    if exog_cols:
        if exog_test is not None and len(exog_test) > 0:
            last_vals = exog_test.iloc[-1:][exog_cols]
        elif exog_train is not None and len(exog_train) > 0:
            last_vals = exog_train.iloc[-1:][exog_cols]
        else:
            last_vals = pd.DataFrame([[0] * len(exog_cols)], columns=exog_cols)

        fut = (
            pd.DataFrame(np.repeat(last_vals.values, max(int(extra_h), 0), axis=0), columns=exog_cols)
            if extra_h > 0 else pd.DataFrame(columns=exog_cols)
        )

        if exog_test is not None and len(exog_test) > 0:
            exog_steps = pd.concat([exog_test.reset_index(drop=True), fut], ignore_index=True)
        else:
            exog_steps = fut if steps > 0 else None

    pred = fit.forecast(steps=steps, exog=exog_steps)
    pred.index = pd.date_range(y_train.index[-1], periods=steps + 1, freq=freq)[1:]
    return pd.DataFrame({"forecast": pred})

# --- Helper: Prophet fit + forecast ---
def fit_predict_prophet(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        extra_h: int,
                        seasonality_mode: str,
                        cps: float,
                        freq: str,
                        exog_cols) -> pd.DataFrame:
    """
    Train Prophet on train_df and forecast len(test_df)+extra_h periods.
    train_df/test_df are indexed by datetime and contain y_col plus optional exog_cols.
    Returns DataFrame with index (future dates) and column 'forecast'.
    """
    # Safety: prophet import / availability
    try:
        from prophet import Prophet  # modern package
    except Exception:
        try:
            from fbprophet import Prophet  # legacy fallback
        except Exception as e:
            raise RuntimeError("Prophet is not installed. Install 'prophet' or 'fbprophet'.") from e

    # Prepare training frame for Prophet
    dfr = train_df.rename(columns={y_col: "y"}).copy()
    dfr["ds"] = dfr.index
    cols = ["ds", "y"] + [c for c in exog_cols if c in dfr.columns]
    dfr = dfr[cols]

    # Build model
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=cps,
    )
    for reg in exog_cols:
        if reg in dfr.columns:
            m.add_regressor(reg)

    # Fit
    m.fit(dfr)

    # Future periods = test length + extra horizon
    periods = len(test_df) + int(extra_h)
    future = m.make_future_dataframe(periods=periods, freq=freq, include_history=False)

    # Carry-forward last known exogenous values (simple baseline)
    for reg in exog_cols:
        if reg in dfr.columns:
            future[reg] = float(dfr[reg].iloc[-1])

    # Predict
    fc = m.predict(future)
    out = fc[["ds", "yhat"]].rename(columns={"ds": "date", "yhat": "forecast"})
    out.set_index("date", inplace=True)
    return out
# -------------------------------
# Train & Forecast
# -------------------------------
with st.spinner("Training model and forecasting…"):
    if model_choice == "Prophet":
        if not HAVE_PROPHET:
            st.error("Prophet is not installed in this environment.")
            st.stop()
        fc = fit_predict_prophet(train, test, future_h, seasonality_mode, cps, freq, exog_candidates)
    else:
        order = (int(p), int(d), int(q))
        seas = (int(P), int(D), int(Q), int(m))
        fc = fit_predict_sarimax(train, test, future_h, order, seas, freq, exog_candidates)

# -------------------------------
# Evaluate on test window
# -------------------------------
forecast_on_test = fc.iloc[: len(test)]
merged = pd.concat([train[[y_col]].assign(split="train"),
                    test[[y_col]].assign(split="test")]).rename(columns={y_col: "actual"})

# Align for metrics
actual = test[y_col].iloc[: len(forecast_on_test)]
pred = forecast_on_test["forecast"].iloc[: len(actual)]

mae = mean_absolute_error(actual, pred)
rmse = rmse_compat(actual, pred)
mape = mape_compat(actual, pred)

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("MAPE", f"{mape:.2f}%")

# -------------------------------
# Interactive charts
# -------------------------------
# 1) Actuals vs Forecast with test shading
fig = go.Figure()
fig.add_trace(go.Scatter(x=merged.index, y=merged["actual"], mode="lines", name="Actual"))
fig.add_trace(go.Scatter(x=fc.index, y=fc["forecast"], mode="lines", name="Forecast"))

fig.add_vrect(
    x0=test.index.min(), x1=test.index.max(),
    fillcolor="LightSalmon", opacity=0.25, line_width=0,
    annotation_text="Test", annotation_position="top left"
)
fig.update_layout(
    title="Actuals vs Forecast",
    xaxis_title="Date",
    yaxis_title=y_col,
    legend_title="Series",
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# 2) Residuals (test)
resid = (actual - pred)
fig2 = px.line(x=actual.index, y=resid, labels={"x": "Date", "y": "Residual (Actual - Forecast)"},
               title="Residuals (Test window)")
st.plotly_chart(fig2, use_container_width=True)



st.subheader("🧠 AI Forecast Explanation")

if st.button("Explain forecast in words"):
    with st.spinner("Generating explanation..."):
        explanation = explain_with_llm(
            model_choice=model_choice,
            freq=freq,
            train=train,
            test=test,
            forecast=fc,
            mae=mae,
            rmse=rmse,
            mape=mape,
            exog_cols=exog_candidates
        )

    st.markdown(explanation)

# -------------------------------
# Download forecast
# -------------------------------
out = fc.copy()
out.index.name = "date"
csv_bytes = out.reset_index().to_csv(index=False).encode()
st.download_button("⬇️ Download forecast CSV", data=csv_bytes, file_name="forecast_output.csv", mime="text/csv")

st.divider()


