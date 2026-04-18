# dashboard_genai_docker.py — Docker-compatible version
# ============================================================
# 🛡️ Real-Time Fraud Detection Dashboard — with GenAI
# ============================================================

import os
import glob
from typing import List

import pandas as pd
import streamlit as st
import altair as alt
import joblib

# ---------------------------------------------------
# CONFIG — Docker paths
# ---------------------------------------------------

DATA_DIR = "/app/fraud_output/alerts"
MODEL_PATH = "/app/models/fraud_model_paysim_rf.pkl"

FEATURES: List[str] = [
    "type", "amount", "oldbalanceOrg",
    "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
]

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Real-Time Fraud Monitoring + GenAI",
    layout="wide",
    page_icon="🛡️",
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>
    .big-title { font-size: 36px; font-weight: 700; margin-bottom: 0; }
    .subtitle { font-size: 16px; color: #888; margin-top: 0; }
    .genai-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
        padding: 20px 24px;
        border-radius: 12px;
        border-left: 4px solid #00d4ff;
        margin: 10px 0;
        font-size: 15px;
        line-height: 1.7;
    }
    .genai-box strong { color: #00d4ff; }
    .genai-label {
        font-size: 12px;
        color: #00d4ff;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown('<div class="big-title">🛡️ Real-Time Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Kafka ➜ Spark ➜ ML Model ➜ GenAI Explanations | Dockerized Pipeline</div>', unsafe_allow_html=True)
st.markdown("")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    # Try environment variable first, then allow manual input
    env_key = os.environ.get("OPENAI_API_KEY", "")
    openai_key = st.text_input(
        "OpenAI API Key",
        value=env_key,
        type="password",
        help="Enter your OpenAI API key or set OPENAI_API_KEY env variable",
        placeholder="sk-...",
    )
    if openai_key:
        st.success("✅ API key set")
    else:
        st.info("Enter your OpenAI key to enable GenAI explanations")

    st.divider()
    st.markdown("""
    **🐳 Docker Mode**  
    All services running via `docker-compose up`
    
    **Services:**
    - ZooKeeper ✅
    - Kafka Broker ✅
    - Spark Consumer ✅
    - Kafka Producer ✅
    - This Dashboard ✅
    """)

# ---------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        return None

# ---------------------------------------------------
# GenAI EXPLANATION
# ---------------------------------------------------

def generate_fraud_explanation(txn_data: dict, shap_contributions: dict = None) -> str:
    if not openai_key:
        return "⚠️ Enter your OpenAI API key in the sidebar to enable GenAI explanations."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
    except ImportError:
        return "⚠️ OpenAI package not installed."

    shap_text = ""
    if shap_contributions:
        sorted_contribs = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        shap_text = "\n".join([
            f"  - {feat}: {val:+.4f} ({'increases' if val > 0 else 'decreases'} fraud risk)"
            for feat, val in sorted_contribs[:6]
        ])
        shap_text = f"\nSHAP Feature Contributions (impact on fraud score):\n{shap_text}"

    prompt = f"""You are a fraud detection analyst AI. A transaction has been flagged by our 
real-time fraud detection system. Analyze the transaction and provide a clear, concise 
explanation that a fraud analyst can understand.

Transaction Details:
- Transaction ID: {txn_data.get('transaction_id', 'N/A')}
- Type: {txn_data.get('type', 'N/A')}
- Amount: ${txn_data.get('amount', 0):,.2f}
- Sender's Original Balance: ${txn_data.get('oldbalanceOrg', 0):,.2f}
- Sender's New Balance: ${txn_data.get('newbalanceOrig', 0):,.2f}
- Receiver's Original Balance: ${txn_data.get('oldbalanceDest', 0):,.2f}
- Receiver's New Balance: ${txn_data.get('newbalanceDest', 0):,.2f}
- ML Fraud Score: {txn_data.get('ml_score', 'N/A')}
- Severity: {txn_data.get('severity', 'N/A')}
- Rule Flags: High Amount={txn_data.get('high_amount_flag', 0)}, Zero Balance Mismatch={txn_data.get('zero_balance_mismatch', 0)}
{shap_text}

Provide your analysis in this format:
1. **Risk Assessment**: One sentence summary of risk level
2. **Key Indicators**: 2-3 bullet points explaining WHY this was flagged  
3. **Pattern Match**: What known fraud pattern does this resemble
4. **Recommended Action**: What should the analyst do next

Keep it concise (under 150 words). Be specific to THIS transaction's numbers."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise fraud detection analyst. Give specific, actionable analysis based on the transaction data provided."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ GenAI error: {str(e)}"


# ---------------------------------------------------
# LOAD ALERTS
# ---------------------------------------------------

def load_alerts() -> pd.DataFrame:
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
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
    df.columns = [c.split("/")[-1] for c in df.columns]

    if {"transaction_id", "timestamp"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["transaction_id", "timestamp"])

    return df

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = load_alerts()

if "severity" not in df.columns:
    df["severity"] = "unknown"
df["severity"] = df["severity"].fillna("unknown").astype(str).str.lower()
valid_sev = {"critical", "high", "medium", "low", "none", "unknown"}
df["severity"] = df["severity"].apply(lambda x: x if x in valid_sev else "unknown")

if df.empty:
    st.warning(
        "⚠ No alerts found yet.\n\n"
        "• Spark consumer and Kafka producer are starting up.\n"
        "• Wait 15-30 seconds, then refresh the page (Ctrl+R / ⌘+R)."
    )
    st.stop()

if "timestamp" in df.columns:
    df["ts"] = pd.to_datetime(df["timestamp"].astype(float), unit="s", errors="coerce")
else:
    df["ts"] = pd.NaT

df = df.sort_values("ts", ascending=False)

# ---------------------------------------------------
# KPIs
# ---------------------------------------------------

st.subheader("📊 Summary")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Alerts", len(df))

if "ml_score" in df.columns:
    col2.metric("Avg ML Score", round(df["ml_score"].mean(), 3))
    col3.metric("High-Risk (≥ 0.9)", int((df["ml_score"] >= 0.9).sum()))
else:
    col2.metric("Avg ML Score", "N/A")
    col3.metric("High-Risk", "N/A")

if "severity" in df.columns:
    col4.metric("🚨 Critical", int((df["severity"] == "critical").sum()))

# ---------------------------------------------------
# GenAI SECTION
# ---------------------------------------------------

st.markdown("---")
st.subheader("🤖 GenAI Fraud Analyst")
st.caption("Select a flagged transaction to get an AI-powered explanation.")

explain_df = df.copy()
if "is_fraud_ml" in explain_df.columns:
    high_risk = explain_df[
        (explain_df["is_fraud_ml"] == 1) | (explain_df["severity"].isin(["critical", "high"]))
    ]
    if not high_risk.empty:
        explain_df = high_risk

genai_col1, genai_col2 = st.columns([1, 2])

with genai_col1:
    options = explain_df["transaction_id"].astype(str).tolist()[:100]
    selected_txn = st.selectbox("Choose a transaction:", options, index=0 if options else None)

    if selected_txn:
        row = explain_df[explain_df["transaction_id"].astype(str) == selected_txn].iloc[0]
        st.markdown(f"""
        **Transaction:** `{row.get('transaction_id', 'N/A')}`  
        **Type:** `{row.get('type', 'N/A')}`  
        **Amount:** `${float(row.get('amount', 0)):,.2f}`  
        **ML Score:** `{float(row.get('ml_score', 0)):.3f}`  
        **Severity:** `{row.get('severity', 'N/A')}`
        """)

with genai_col2:
    if selected_txn:
        model = load_model()
        shap_contribs = None

        if model is not None:
            try:
                import shap
                background = df[FEATURES].dropna().sample(min(100, len(df)), random_state=42)

                def predict_fn(X_array):
                    X_df = pd.DataFrame(X_array, columns=FEATURES)
                    return model.predict_proba(X_df)[:, 1]

                explainer = shap.KernelExplainer(predict_fn, background.values)
                x_row = row[FEATURES].values.reshape(1, -1)
                sv = explainer.shap_values(x_row)[0]
                shap_contribs = dict(zip(FEATURES, sv))
            except Exception:
                shap_contribs = None

        if st.button("🧠 Generate AI Explanation", type="primary", use_container_width=True):
            txn_data = row.to_dict()
            with st.spinner("🔄 AI is analyzing this transaction..."):
                explanation = generate_fraud_explanation(txn_data, shap_contribs)

            st.markdown('<div class="genai-label">🤖 AI Fraud Analysis</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="genai-box">{explanation}</div>', unsafe_allow_html=True)

            if shap_contribs:
                with st.expander("📊 SHAP Feature Contributions"):
                    contrib_series = pd.Series(shap_contribs).sort_values(key=abs, ascending=False)
                    st.bar_chart(contrib_series)

# ---------------------------------------------------
# ALERTS TABLE
# ---------------------------------------------------

st.markdown("---")
st.subheader("🚨 Latest Fraud Alerts")

cols_to_show = [
    c for c in [
        "transaction_id", "ts", "type", "amount",
        "ml_score", "severity", "is_fraud_ml",
        "rule_flag", "is_fraud_final",
    ] if c in df.columns
]

latest_limit = st.slider("Recent alerts to show:", 20, 500, 100)
df_latest = df.head(latest_limit)[cols_to_show]

def color_severity(val):
    if val == "critical": return "background-color: #ff4b4b; color: white;"
    if val == "high": return "background-color: #ffa600; color: black;"
    if val == "medium": return "background-color: #ffd480; color: black;"
    if val == "low": return "background-color: #d2f5d2; color: black;"
    return ""

if "severity" in df_latest.columns:
    styled = df_latest.style.applymap(color_severity, subset=["severity"])
    st.dataframe(styled, use_container_width=True, height=350)
else:
    st.dataframe(df_latest, use_container_width=True, height=350)

# ---------------------------------------------------
# CHARTS
# ---------------------------------------------------

st.subheader("📈 Visual Analytics")
c1, c2 = st.columns(2)

if "ml_score" in df.columns:
    with c1:
        st.markdown("**Fraud Score Distribution**")
        score_chart = (
            alt.Chart(df).mark_bar()
            .encode(
                alt.X("ml_score:Q", bin=alt.Bin(maxbins=30), title="ML Fraud Score"),
                y=alt.Y("count():Q", title="Number of alerts"),
                color=alt.Color("severity:N", legend=alt.Legend(title="Severity")),
            ).properties(height=300)
        )
        st.altair_chart(score_chart, use_container_width=True)

if "type" in df.columns:
    with c2:
        st.markdown("**Alerts by Transaction Type**")
        type_chart = (
            alt.Chart(df).mark_bar()
            .encode(
                x=alt.X("type:N", title="Transaction type"),
                y=alt.Y("count():Q", title="Number of alerts"),
                color=alt.Color("severity:N", legend=None),
            ).properties(height=300)
        )
        st.altair_chart(type_chart, use_container_width=True)

if "ts" in df.columns and "ml_score" in df.columns:
    st.markdown("**Fraud Score Trend**")
    df_time = df.dropna(subset=["ts"]).head(300)
    if not df_time.empty:
        line_chart = (
            alt.Chart(df_time).mark_line(point=True)
            .encode(
                x=alt.X("ts:T", title="Time"),
                y=alt.Y("ml_score:Q", title="ML score"),
                color=alt.Color("severity:N", title="Severity"),
                tooltip=["transaction_id", "type", "amount", "ml_score", "severity"],
            ).properties(height=300)
        )
        st.altair_chart(line_chart, use_container_width=True)

# ---------------------------------------------------
# INFO
# ---------------------------------------------------

with st.expander("ℹ️ How to read this dashboard"):
    st.markdown("""
**Pipeline:** Kafka ➜ Spark ➜ ML Model ➜ Dashboard

**Columns:** `ml_score` = fraud probability (0-1), `severity` = alert urgency,
`is_fraud_ml` = ML flagged, `rule_flag` = business rule triggered,
`is_fraud_final` = treated as fraud (ML OR rules)

**🤖 GenAI:** Select a transaction → click "Generate AI Explanation" → get plain-English analysis powered by GPT-4o-mini + SHAP values.
""")

st.caption("Real-Time Fraud Detection Platform | Nitish Bhattad & Sowmika Dinesh | DSC 550")
