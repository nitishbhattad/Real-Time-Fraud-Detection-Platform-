# kafka_paysim_producer_docker.py — Docker-compatible version

import json
import time
import random
import os

import pandas as pd
from kafka import KafkaProducer

# ============================================
# Config from environment
# ============================================

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
DATA_PATH = "/app/data/paysim.csv"
TOPIC = "transactions"
P_FRAUD = 0.20  # 20% fraud for demo visibility

# ============================================
# Load PaySim dataset
# ============================================

print(f"📄 Loading PaySim dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

required_cols = [
    "type", "amount", "oldbalanceOrg",
    "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"❌ Missing columns: {missing}")

df = df[required_cols].copy()

if len(df) > 500_000:
    df = df.sample(500_000, random_state=42).reset_index(drop=True)

fraud_df = df[df["isFraud"] == 1].reset_index(drop=True)
legit_df = df[df["isFraud"] == 0].reset_index(drop=True)

print(f"✅ Loaded {len(df)} rows ({len(fraud_df)} fraud, {len(legit_df)} legit)")

# ============================================
# Connect to Kafka (with retry)
# ============================================

print(f"🔌 Connecting to Kafka at {KAFKA_BOOTSTRAP}...")

for attempt in range(30):
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        print(f"✅ Connected to Kafka. Topic = '{TOPIC}'")
        break
    except Exception as e:
        print(f"⏳ Waiting for Kafka (attempt {attempt+1}/30)... {e}")
        time.sleep(2)
else:
    raise ConnectionError("❌ Could not connect to Kafka after 30 attempts")

# ============================================
# Stream transactions
# ============================================

i = 0
try:
    while True:
        if random.random() < P_FRAUD:
            row = fraud_df.iloc[random.randint(0, len(fraud_df) - 1)]
        else:
            row = legit_df.iloc[random.randint(0, len(legit_df) - 1)]

        record = {
            "transaction_id": f"txn_{i}",
            "timestamp": str(time.time()),
            "type": row["type"],
            "amount": float(row["amount"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "newbalanceOrig": float(row["newbalanceOrig"]),
            "oldbalanceDest": float(row["oldbalanceDest"]),
            "newbalanceDest": float(row["newbalanceDest"]),
            "is_fraud_true": int(row["isFraud"]),
        }

        producer.send(TOPIC, record)
        print(f"📤 Sent: txn_{i} | type={record['type']} | amount=${record['amount']:.2f} | fraud={record['is_fraud_true']}")

        i += 1
        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n🛑 Stopping producer")
finally:
    producer.flush()
    producer.close()
    print("✅ Producer closed.")
