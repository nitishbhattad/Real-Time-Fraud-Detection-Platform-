# ~/Desktop/kafka-fraud-detection/streamlit_app.py
import os, glob, signal, subprocess
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

FRAUD_DIR = os.path.expanduser("~/Desktop/kafka-fraud-detection/fraud_output/alerts")
PRODUCER_CMD = ["python", "kafka_producer.py"]
CSV_COLUMNS = ["transaction_id","user_id","amount","location","device_type","merchant","timestamp","is_fraud","rule_score"]
REFRESH_MS = 2000

st.set_page_config(page_title="Fraud Demo", layout="wide")
st.title("💳 Real-time Fraud Detection — Live Demo")

with st.sidebar:
    st.header("Run order (in terminals)")
    st.markdown("""
1. **ZooKeeper**  
   `cd ~/kafka && bin/zookeeper-server-start.sh config/zookeeper.properties`
2. **Kafka broker**  
   `cd ~/kafka && bin/kafka-server-start.sh config/server.properties`
3. **Spark consumer**  
   `cd ~/Desktop/kafka-fraud-detection && source venv311/bin/activate`  
   `python spark_streaming_consumer.py`
4. **Here:** Start the Producer.
""")
    st.caption(f"Topic: `transactions` • Alerts dir: `{FRAUD_DIR}`")

if "producer" not in st.session_state: st.session_state.producer = None
c1,c2 = st.columns(2)
if c1.button("▶️ Start Producer", type="primary"):
    if st.session_state.producer and st.session_state.producer.poll() is None:
        st.success("Producer already running.")
    else:
        try:
            st.session_state.producer = subprocess.Popen(PRODUCER_CMD, cwd=os.getcwd())
            st.success("Producer started.")
        except Exception as e:
            st.error(f"Failed to start producer: {e}")
if c2.button("⏹ Stop Producer"):
    p = st.session_state.producer
    if p and p.poll() is None:
        try:
            os.kill(p.pid, signal.SIGTERM)
            st.success("Producer stopped.")
        except Exception as e:
            st.error(f"Failed to stop producer: {e}")
    else:
        st.info("Producer not running.")

@st.cache_data(ttl=1.0)
def load_alerts():
    if not os.path.isdir(FRAUD_DIR):
        return pd.DataFrame(columns=CSV_COLUMNS)
    files = sorted(glob.glob(os.path.join(FRAUD_DIR, "*.csv")))
    if not files:
        return pd.DataFrame(columns=CSV_COLUMNS)
    dfs = []
    for f in files[-80:]:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass
    if not dfs: return pd.DataFrame(columns=CSV_COLUMNS)
    out = pd.concat(dfs, ignore_index=True)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
        out = out.dropna(subset=["timestamp"])
    return out

st_autorefresh(interval=REFRESH_MS, key="auto")
df = load_alerts()

k1,k2,k3,k4 = st.columns(4)
now = datetime.now(timezone.utc)
k1.metric("Fraud alerts (loaded)", f"{len(df):,}")
k2.metric("Alerts in last min", f"{len(df[df['timestamp']>now-pd.Timedelta(minutes=1)]):,}" if len(df) else "0")
k3.metric("Avg amount (alerts)", f"{df['amount'].mean():,.2f}" if len(df) else "—")
k4.metric("Last alert age (s)", f"{(now - df['timestamp'].max()).total_seconds():.1f}" if len(df) else "—")

st.divider()
left,right = st.columns([1.3,1])
with left:
    st.subheader("Recent fraud alerts")
    if len(df):
        st.dataframe(
            df.sort_values("timestamp", ascending=False)
              .loc[:, ["timestamp","transaction_id","user_id","amount","merchant","location","device_type","rule_score"]]
              .head(60),
            use_container_width=True, height=420
        )
    else:
        st.info("No alerts yet. Start the Producer and keep the Spark consumer running.")
with right:
    st.subheader("Frauds per minute")
    if len(df):
        per_min = (df.set_index("timestamp").sort_index().resample("1min")["transaction_id"].count())
        st.line_chart(per_min)
    else:
        st.line_chart(pd.Series(dtype="int"))

st.subheader("Breakdowns")
b1,b2 = st.columns(2)
with b1:
    st.caption("Top merchants (alerts)")
    if len(df): st.bar_chart(df.groupby("merchant")["transaction_id"].count().sort_values(ascending=False).head(10))
    else:       st.bar_chart(pd.Series(dtype="int"))
with b2:
    st.caption("Top locations (alerts)")
    if len(df): st.bar_chart(df.groupby("location")    ["transaction_id"].count().sort_values(ascending=False).head(10))
    else:       st.bar_chart(pd.Series(dtype="int"))

st.caption("This dashboard reads the CSV alerts written by Spark to the alerts folder.")
