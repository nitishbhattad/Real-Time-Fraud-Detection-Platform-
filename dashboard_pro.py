# ~/Desktop/kafka-fraud-detection/dashboard_pro.py
# ============================================================
# 🛡️ Real-Time Fraud Detection Dashboard — Production Grade
# ============================================================
# Features: Database-backed, Live Accuracy Tracker, GenAI
# Run:  streamlit run dashboard_pro.py
# ============================================================

import os
import sys
import json
from typing import List

import pandas as pd
import streamlit as st
import altair as alt
import joblib

# Add project to path
BASE_DIR = "/Users/nitishbhattad/Desktop/kafka-fraud-detection"
sys.path.insert(0, BASE_DIR)

from database import init_db, get_alerts_dataframe, get_live_performance, cache_explanation, get_cached_explanation

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model_paysim_rf.pkl")

FEATURES: List[str] = [
    "type", "amount", "oldbalanceOrg",
    "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
]

# Initialize database
init_db()

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection Platform",
    layout="wide",
    page_icon="🛡️",
)

# ---------------------------------------------------
# CSS
# ---------------------------------------------------

st.markdown("""
<style>
    .big-title { font-size: 36px; font-weight: 700; margin-bottom: 0; }
    .subtitle { font-size: 16px; color: #888; margin-top: 0; }
    .genai-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0; padding: 20px 24px; border-radius: 12px;
        border-left: 4px solid #00d4ff; margin: 10px 0; font-size: 15px; line-height: 1.7;
    }
    .genai-box strong { color: #00d4ff; }
    .genai-label {
        font-size: 12px; color: #00d4ff; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
    }
    .perf-good { color: #22c55e; font-size: 28px; font-weight: 700; }
    .perf-warn { color: #f59e0b; font-size: 28px; font-weight: 700; }
    .perf-bad { color: #ef4444; font-size: 28px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown('<div class="big-title">🛡️ Real-Time Fraud Detection Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Kafka ➜ Spark ➜ ML + Rules ➜ Database ➜ GenAI Explanations | Production Architecture</div>', unsafe_allow_html=True)
st.markdown("")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    env_key = os.environ.get("OPENAI_API_KEY", "")
    openai_key = st.text_input("OpenAI API Key", value=env_key, type="password", placeholder="sk-...")
    if openai_key:
        st.success("✅ API key set")
    else:
        st.info("Enter OpenAI key for GenAI")

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh (every 5s)", value=True)
    if auto_refresh:
        st.markdown("🔄 Dashboard refreshes automatically")

    st.divider()
    st.markdown("""
    **Architecture:**
    - Kafka (message broker)
    - Spark (ML scoring)
    - SQLite/PostgreSQL (storage)
    - FastAPI (REST endpoint)
    - Streamlit (this dashboard)
    - OpenAI GPT (explanations)
    """)

# Auto-refresh
if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=5000, key="auto")
    except ImportError:
        st.cache_data.clear()

# ---------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

# ---------------------------------------------------
# GenAI FUNCTION
# ---------------------------------------------------

def generate_fraud_explanation(txn_data: dict, shap_contributions: dict = None) -> str:
    if not openai_key:
        return "⚠️ Enter your OpenAI API key in the sidebar."

    # Check cache first
    txn_id = txn_data.get("transaction_id", "")
    cached = get_cached_explanation(txn_id)
    if cached:
        return cached

    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
    except ImportError:
        return "⚠️ Run: pip install openai"

    shap_text = ""
    if shap_contributions:
        sorted_contribs = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        shap_text = "\n".join([
            f"  - {feat}: {val:+.4f} ({'increases' if val > 0 else 'decreases'} fraud risk)"
            for feat, val in sorted_contribs[:6]
        ])
        shap_text = f"\nSHAP Feature Contributions:\n{shap_text}"

    prompt = f"""You are a fraud detection analyst AI. Analyze this flagged transaction concisely.

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
{shap_text}

Format:
1. **Risk Assessment**: One sentence
2. **Key Indicators**: 2-3 bullet points
3. **Pattern Match**: What fraud pattern this resembles
4. **Recommended Action**: Next steps

Under 150 words. Be specific to this transaction."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise fraud analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300, temperature=0.3,
        )
        explanation = response.choices[0].message.content

        # Cache it
        shap_json = json.dumps(shap_contributions) if shap_contributions else None
        cache_explanation(txn_id, explanation, shap_json)

        return explanation
    except Exception as e:
        return f"⚠️ GenAI error: {str(e)}"

# ---------------------------------------------------
# LOAD DATA FROM DATABASE
# ---------------------------------------------------

df = get_alerts_dataframe(limit=5000)

if df.empty:
    st.warning("⚠ No alerts in database yet. Run the migration or start the pipeline.")
    st.code("python database.py migrate    # Import existing CSV alerts\n# OR start the live pipeline:\n# Terminal 1: ZooKeeper\n# Terminal 2: Kafka\n# Terminal 3: python spark_streaming_consumer_db.py\n# Terminal 4: python kafka_paysim_producer.py")
    st.stop()

# Severity cleanup
if "severity" not in df.columns:
    df["severity"] = "unknown"
df["severity"] = df["severity"].fillna("unknown").astype(str).str.lower()

# Timestamps
if "timestamp" in df.columns:
    df["ts"] = pd.to_datetime(df["timestamp"].astype(float), unit="s", errors="coerce")
else:
    df["ts"] = pd.NaT

df = df.sort_values("ts", ascending=False)

# ---------------------------------------------------
# TAB LAYOUT
# ---------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🤖 GenAI Analyst", "📈 Live Performance", "🔍 Search"
])

# ===================================================
# TAB 1: OVERVIEW
# ===================================================

# ===================================================
# TAB 1: OVERVIEW
# ===================================================

with tab1:
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Alerts", f"{len(df):,}")

    if "ml_score" in df.columns:
        col2.metric("Avg ML Score", f"{df['ml_score'].mean():.3f}")
        col3.metric("High-Risk (≥0.9)", int((df["ml_score"] >= 0.9).sum()))
    
    if "amount" in df.columns:
        total_fraud_amount = df["amount"].sum()
        col4.metric("💰 Fraud $ Detected", f"${total_fraud_amount:,.0f}")

    critical_count = int((df["severity"] == "critical").sum()) if "severity" in df.columns else 0
    col5.metric("🚨 Critical", critical_count)

    # Alerts table
    st.subheader("🚨 Latest Fraud Alerts")
    cols_to_show = [c for c in ["transaction_id", "ts", "type", "amount", "ml_score", "severity", "is_fraud_ml", "rule_flag", "is_fraud_final", "is_fraud_true"] if c in df.columns]
    latest_limit = st.slider("Show:", 20, 500, 100, key="overview_slider")

    display_df = df.head(latest_limit)[cols_to_show].copy()
    display_df = display_df.rename(columns={
        "transaction_id": "Transaction ID",
        "ts": "Time",
        "type": "Transaction Type",
        "amount": "Amount",
        "ml_score": "Fraud Score",
        "severity": "Risk Level",
        "is_fraud_ml": "AI Flagged",
        "rule_flag": "Rule Triggered",
        "is_fraud_final": "Fraud Alert",
        "is_fraud_true": "Confirmed Fraud",
    })
    if "Amount" in display_df.columns:
        display_df["Amount"] = display_df["Amount"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    if "Risk Level" in display_df.columns:
        display_df["Risk Level"] = display_df["Risk Level"].str.upper()
    for col in ["AI Flagged", "Rule Triggered", "Fraud Alert", "Confirmed Fraud"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: "✅ Yes" if x == 1 or x == 1.0 else ("❌ No" if x == 0 or x == 0.0 else "N/A")
            )
    st.dataframe(display_df, use_container_width=True, height=350)

    # Charts
    c1, c2 = st.columns(2)
    if "ml_score" in df.columns:
        with c1:
            st.markdown("**Fraud Score Distribution**")
            chart = alt.Chart(df).mark_bar().encode(
                alt.X("ml_score:Q", bin=alt.Bin(maxbins=30), title="ML Score"),
                y="count()", color="severity:N"
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

    if "type" in df.columns:
        with c2:
            st.markdown("**Alerts by Type**")
            chart = alt.Chart(df).mark_bar().encode(
                x="type:N", y="count()", color="severity:N"
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

# ===================================================
# TAB 2: GenAI ANALYST
# ===================================================

with tab2:
    st.subheader("🤖 GenAI Fraud Analyst")
    st.caption("AI-powered transaction analysis using GPT-4o-mini + SHAP explainability")

    explain_df = df.copy()
    if "is_fraud_ml" in explain_df.columns:
        high_risk = explain_df[explain_df["is_fraud_ml"] == 1]
        if not high_risk.empty:
            explain_df = high_risk

    g1, g2 = st.columns([1, 2])

    with g1:
        options = explain_df["transaction_id"].astype(str).tolist()[:100]
        selected = st.selectbox("Transaction:", options, index=0 if options else None)
        if selected:
            row = explain_df[explain_df["transaction_id"].astype(str) == selected].iloc[0]
            st.markdown(f"""
            **ID:** `{row.get('transaction_id')}`  
            **Type:** `{row.get('type')}`  
            **Amount:** `${float(row.get('amount', 0)):,.2f}`  
            **ML Score:** `{float(row.get('ml_score', 0)):.3f}`  
            **Severity:** `{row.get('severity')}`  
            **Ground Truth:** `{'FRAUD' if row.get('is_fraud_true') == 1 else 'LEGIT' if row.get('is_fraud_true') == 0 else 'N/A'}`
            """)

    with g2:
        if selected:
            shap_contribs = None
            if st.button("🧠 Generate AI Explanation", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing..."):
                    explanation = generate_fraud_explanation(row.to_dict(), shap_contribs)
                st.session_state["last_explanation"] = explanation
                st.session_state["last_explained_txn"] = selected

            if "last_explanation" in st.session_state:
                st.markdown(f"**Explanation for:** `{st.session_state.get('last_explained_txn', '')}`")
                st.markdown('<div class="genai-label">🤖 AI Fraud Analysis</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="genai-box">{st.session_state["last_explanation"]}</div>', unsafe_allow_html=True)


# ===================================================
# TAB 3: LIVE PERFORMANCE
# ===================================================

with tab3:
    st.subheader("📈 Live Model Performance")
    st.caption("Real-time accuracy metrics — comparing ML predictions vs. ground truth labels")

    perf = get_live_performance()

    if perf is None:
        st.info("No ground truth data available yet. Start the pipeline with `kafka_paysim_producer.py` which sends ground truth labels.")
    else:
        # Big metrics
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Precision", f"{perf['precision']:.1%}")
        p2.metric("Recall", f"{perf['recall']:.1%}")
        p3.metric("F1 Score", f"{perf['f1_score']:.1%}")
        p4.metric("Accuracy", f"{perf['accuracy']:.1%}")

        # Confusion matrix counts
        st.markdown("---")
        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.metric("✅ True Positives", perf["true_positives"])
        cm2.metric("⚠️ False Positives", perf["false_positives"])
        cm3.metric("❌ False Negatives", perf["false_negatives"])
        cm4.metric("✅ True Negatives", perf["true_negatives"])

        st.markdown(f"**Total scored:** {perf['total_predictions']:,} transactions")

        # Performance interpretation
        st.markdown("---")
        if perf["precision"] >= 0.7 and perf["recall"] >= 0.7:
            st.success("🟢 Model performing well — high precision and recall")
        elif perf["precision"] >= 0.5:
            st.warning("🟡 Model performance is acceptable but could improve")
        else:
            st.error("🔴 Model may need retraining — low precision detected")

    # Fraud rate over time
    if "ts" in df.columns and "is_fraud_true" in df.columns:
        st.markdown("---")
        st.markdown("**Fraud Detection Over Time**")
        df_time = df.dropna(subset=["ts"]).copy()
        if not df_time.empty and "is_fraud_true" in df_time.columns:
            df_time["minute"] = df_time["ts"].dt.floor("1min")
            time_agg = df_time.groupby("minute").agg(
                total=("transaction_id", "count"),
                true_fraud=("is_fraud_true", "sum"),
                ml_flagged=("is_fraud_ml", "sum"),
            ).reset_index()

            chart = alt.Chart(time_agg).transform_fold(
                ["true_fraud", "ml_flagged"], as_=["Metric", "Count"]
            ).mark_line(point=True).encode(
                x=alt.X("minute:T", title="Time"),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color("Metric:N", scale=alt.Scale(
                    domain=["true_fraud", "ml_flagged"],
                    range=["#ef4444", "#3b82f6"]
                )),
                tooltip=["minute:T", "Metric:N", "Count:Q"]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)


# ===================================================
# TAB 4: SEARCH
# ===================================================

with tab4:
    st.subheader("🔍 Transaction Search")

    s1, s2, s3 = st.columns(3)

    with s1:
        search_id = st.text_input("Transaction ID", placeholder="txn_123")
    with s2:
        min_score = st.slider("Min ML Score", 0.0, 1.0, 0.0, 0.05)
    with s3:
        type_filter = st.multiselect("Transaction Type", df["type"].unique().tolist() if "type" in df.columns else [])

    # Apply filters
    filtered = df.copy()
    if search_id:
        filtered = filtered[filtered["transaction_id"].astype(str).str.contains(search_id, case=False)]
    if min_score > 0 and "ml_score" in filtered.columns:
        filtered = filtered[filtered["ml_score"] >= min_score]
    if type_filter and "type" in filtered.columns:
        filtered = filtered[filtered["type"].isin(type_filter)]

    st.markdown(f"**{len(filtered):,}** transactions match your filters")

    cols = [c for c in ["transaction_id", "ts", "type", "amount", "ml_score", "severity", "is_fraud_ml", "is_fraud_true"] if c in filtered.columns]
    st.dataframe(filtered[cols].head(200), use_container_width=True, height=400)

    # Amount stats
    if "amount" in filtered.columns and len(filtered) > 0:
        a1, a2, a3 = st.columns(3)
        a1.metric("Total Amount", f"${filtered['amount'].sum():,.0f}")
        a2.metric("Avg Amount", f"${filtered['amount'].mean():,.0f}")
        a3.metric("Max Amount", f"${filtered['amount'].max():,.0f}")


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")
st.caption("Real-Time Fraud Detection Platform | Nitish Bhattad & Sowmika Dinesh | DSC 550 | Database-backed • FastAPI • GenAI")
