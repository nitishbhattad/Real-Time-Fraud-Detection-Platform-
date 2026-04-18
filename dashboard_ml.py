# ~/Desktop/kafka-fraud-detection/dashboard_ml.py

import os
import glob
from typing import List

import pandas as pd
import streamlit as st
import altair as alt
import joblib

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

BASE_DIR = "/Users/nitishbhattad/Desktop/kafka-fraud-detection"

DATA_DIR = os.path.join(BASE_DIR, "fraud_output", "alerts")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model_paysim_rf.pkl")

FEATURES: List[str] = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

st.set_page_config(
    page_title="Real-Time Fraud Monitoring",
    layout="wide",
    page_icon="🛡️",
)

# ---------------------------------------------------
# Model loader (single, clean version)
# ---------------------------------------------------

@st.cache_resource
def load_model():
    """
    Load the trained RandomForest pipeline for explainability.
    Cached so we load it only once per session.
    """
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        # Ensure sklearn is available in THIS process
        import sklearn  # type: ignore

        st.write(f"Loaded sklearn version: {sklearn.__version__}")
        st.write(f"Using Python executable: {os.sys.executable}")

        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.warning(f"Could not load model from {MODEL_PATH}: {e}")
        return None


# ---------------------------------------------------
# Header / title
# ---------------------------------------------------

st.markdown(
    """
    <style>
    .big-title {
        font-size: 38px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="big-title">🛡️ Real-Time Fraud Detection Dashboard</div>',
    unsafe_allow_html=True,
)
st.caption("Live fraud alerts from Kafka ➜ Spark ➜ Machine Learning model")


# ---------------------------------------------------
# Helper: load alerts safely
# ---------------------------------------------------

def load_alerts() -> pd.DataFrame:
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        # Skip 0-byte or tiny files
        if os.path.getsize(f) < 10:
            continue
        try:
            df_piece = pd.read_csv(f)
            if not df_piece.empty:
                dfs.append(df_piece)
        except pd.errors.EmptyDataError:
            continue

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Clean Spark-style column names like "path/to/col"
    df.columns = [c.split("/")[-1] for c in df.columns]

    # Remove duplicates if we have transaction_id + timestamp
    if {"transaction_id", "timestamp"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["transaction_id", "timestamp"])

    return df


# ---------------------------------------------------
# Load data
# ---------------------------------------------------

df = load_alerts()

# ---- Severity patch (make sure Altair is happy) ----
if "severity" not in df.columns:
    df["severity"] = "unknown"

# Normalize and clean up severity values
df["severity"] = (
    df["severity"]
    .fillna("unknown")
    .astype(str)
    .str.lower()
)

valid_sev = {"critical", "high", "medium", "low", "none", "unknown"}
df["severity"] = df["severity"].apply(lambda x: x if x in valid_sev else "unknown")
# -----------------------------------------------------

if df.empty:
    st.warning(
        "⚠ No alerts found yet.\n\n"
        "• Make sure Spark streaming consumer is running.\n"
        "• Make sure `kafka_paysim_producer.py` is sending transactions.\n"
        "• Wait a few seconds, then refresh the page (Ctrl+R / ⌘+R)."
    )
    st.stop()

# Convert timestamp to sortable / time-series friendly column
if "timestamp" in df.columns:
    # your timestamps look like floating epoch seconds as strings
    df["ts"] = pd.to_datetime(
        df["timestamp"].astype(float), unit="s", errors="coerce"
    )
else:
    df["ts"] = pd.NaT

# Sort newest first
df = df.sort_values("ts", ascending=False)


# ---------------------------------------------------
# Summary KPIs
# ---------------------------------------------------

st.subheader("📊 Summary")

col1, col2, col3, col4 = st.columns(4)

# Total alerts
col1.metric("Total Alerts", len(df))

# Average ML score
if "ml_score" in df.columns:
    col2.metric("Average ML Fraud Score", round(df["ml_score"].mean(), 3))
    high_risk_count = int((df["ml_score"] >= 0.9).sum())
    col3.metric("High-Risk ML Alerts (score ≥ 0.9)", high_risk_count)
else:
    col2.metric("Average ML Fraud Score", "N/A")
    col3.metric("High-Risk ML Alerts", "N/A")

# Rule-based / severity
if "severity" in df.columns:
    critical_count = int((df["severity"] == "critical").sum())
    col4.metric("🚨 Critical Alerts", critical_count)
elif "rule_flag" in df.columns:
    col4.metric("Rule-Based Flags", int(df["rule_flag"].sum()))
else:
    col4.metric("Rule-Based Flags", "N/A")

# Small hint bar
st.info("Tip: Refresh the page to see new incoming alerts.")


# ---------------------------------------------------
# Latest alerts table
# ---------------------------------------------------

st.subheader("🚨 Latest Fraud Alerts")

cols_to_show = [
    c
    for c in [
        "transaction_id",
        "ts",
        "type",
        "amount",
        "ml_score",
        "severity",
        "is_fraud_ml",
        "rule_flag",
        "is_fraud_final",
    ]
    if c in df.columns
]

latest_limit = st.slider("How many recent alerts to show?", 20, 500, 100)
df_latest = df.head(latest_limit)[cols_to_show]


def color_severity(val):
    """Color cells by severity to be friendly for non-technical users."""
    if val == "critical":
        return "background-color: #ff4b4b; color: white;"
    if val == "high":
        return "background-color: #ffa600; color: black;"
    if val == "medium":
        return "background-color: #ffd480; color: black;"
    if val == "low":
        return "background-color: #d2f5d2; color: black;"
    return ""


if "severity" in df_latest.columns:
    styled = df_latest.style.applymap(color_severity, subset=["severity"])
    st.dataframe(styled, use_container_width=True, height=350)
else:
    st.dataframe(df_latest, use_container_width=True, height=350)


# ---------------------------------------------------
# Charts section
# ---------------------------------------------------

st.subheader("📈 Visual Analytics")

c1, c2 = st.columns(2)

# 1) Fraud score distribution (histogram)
if "ml_score" in df.columns:
    with c1:
        st.markdown("**Fraud Score Distribution**")
        score_chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                alt.X("ml_score:Q", bin=alt.Bin(maxbins=30), title="ML Fraud Score"),
                y=alt.Y("count():Q", title="Number of alerts"),
                color=alt.Color("severity:N", legend=alt.Legend(title="Severity")),
            )
            .properties(height=300)
        )
        st.altair_chart(score_chart, use_container_width=True)

# 2) Alerts by transaction type
if "type" in df.columns:
    with c2:
        st.markdown("**Alerts by Transaction Type**")
        type_chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("type:N", title="Transaction type"),
                y=alt.Y("count():Q", title="Number of alerts"),
                color=alt.Color("severity:N", legend=None),
            )
            .properties(height=300)
        )
        st.altair_chart(type_chart, use_container_width=True)

# 3) Time-series of scores
if "ts" in df.columns and "ml_score" in df.columns:
    st.markdown("**Fraud Score Trend (recent alerts)**")
    df_time = df.dropna(subset=["ts"]).head(300)
    if not df_time.empty:
        line_chart = (
            alt.Chart(df_time)
            .mark_line(point=True)
            .encode(
                x=alt.X("ts:T", title="Time"),
                y=alt.Y("ml_score:Q", title="ML score"),
                color=alt.Color("severity:N", title="Severity"),
                tooltip=["transaction_id", "type", "amount", "ml_score", "severity"],
            )
            .properties(height=300)
        )
        st.altair_chart(line_chart, use_container_width=True)


# ---------------------------------------------------
# ML explainability (SHAP)
# ---------------------------------------------------

st.subheader("🧠 Why did the model flag this transaction? (Explainability)")

model = load_model()
if model is None:
    st.info("Model file not found or could not be loaded. Explainability disabled.")
else:
    # Try to import shap
    try:
        import shap  # type: ignore
        import sklearn  # noqa: F401
    except ImportError:
        st.info(
            "To enable explainability, install SHAP in your environment:\n"
            "`pip install shap`"
        )
        model = None

if model is not None:
    # Prefer transactions that ML actually flagged or that we marked as critical
    explain_candidates = df.copy()
    if "is_fraud_ml" in explain_candidates.columns:
        explain_candidates = explain_candidates[
            (explain_candidates["is_fraud_ml"] == 1)
            | (explain_candidates["severity"] == "critical")
        ]

    # Fallback if that filter is empty
    if explain_candidates.empty:
        explain_candidates = df.tail(100)

    # Selection list
    options = explain_candidates["transaction_id"].astype(str).tolist()
    selected_txn = st.selectbox(
        "Choose a transaction to explain:",
        options,
        index=0 if options else None,
    )

    if selected_txn:
        row = explain_candidates[
            explain_candidates["transaction_id"].astype(str) == selected_txn
        ].iloc[0]

        ml_score_val = float(row.get("ml_score", 0.0))

        st.markdown(
            f"""
            **Transaction `{row['transaction_id']}`**

            - Type: `{row.get('type')}`
            - Amount: `{float(row.get('amount', 0.0)):.2f}`
            - ML Score: `{ml_score_val:.3f}`
            - Severity: `{row.get('severity', 'n/a')}`
            """
        )

        # Background sample for SHAP (small for speed)
        try:
            background = (
                df[FEATURES]
                .dropna()
                .sample(min(200, len(df)), random_state=42)
            )
        except KeyError:
            st.warning(
                "Explainability disabled: some required features "
                f"are missing from the alerts: {FEATURES}"
            )
        else:
            # Prediction function on original feature space
            def predict_proba(X_array):
                X_df = pd.DataFrame(X_array, columns=FEATURES)
                return model.predict_proba(X_df)[:, 1]

            try:
                import shap  # type: ignore

                # KernelExplainer works with any sklearn Pipeline
                explainer = shap.KernelExplainer(
                    predict_proba,
                    background.values,
                )

                x = row[FEATURES].values.reshape(1, -1)
                shap_values = explainer.shap_values(x)[0]  # (n_features,)

                contrib = pd.Series(shap_values, index=FEATURES)
                contrib = contrib.reindex(
                    contrib.abs().sort_values(ascending=False).index
                )

                st.markdown(
                    "**Top feature contributions to fraud score (SHAP values)**"
                )
                st.bar_chart(contrib)

                with st.expander("View raw SHAP values"):
                    st.write(contrib.to_frame("shap_value"))

            except Exception as e:
                st.warning(
                    "Explainability could not be computed right now. "
                    f"Technical detail: {e}"
                )


# ---------------------------------------------------
# Simple explanation for non-technical people
# ---------------------------------------------------

with st.expander("ℹ️ How to read this dashboard"):
    st.markdown(
        """
**What are you seeing?**

- Each row in the table is a **suspicious transaction** (an alert).
- This data comes in **real time** from a streaming pipeline:
  **Kafka ➜ Spark ➜ Machine Learning model ➜ Dashboard**.

**Key columns :**

- `amount` → how much money was moved  
- `ml_score` → how suspicious the transaction looks to the ML model (0 to 1)  
- `severity` → how urgent the alert is (`critical`, `high`, `medium`, `low`)  
- `is_fraud_ml = 1` → the ML model thinks this is fraud  
- `rule_flag = 1` → a simple business rule was triggered  
  (for example: very large amount, or strange balance pattern)  
- `is_fraud_final = 1` → treated as a **fraud alert**  
  (either ML **OR** rules flagged it)
"""
    )
