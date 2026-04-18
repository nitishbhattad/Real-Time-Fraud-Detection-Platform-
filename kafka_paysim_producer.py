# ~/Desktop/kafka-fraud-detection/kafka_paysim_producer.py

import json
import time
import random

import pandas as pd
from kafka import KafkaProducer

# ============================================
# 1. Load PaySim dataset
# ============================================

DATA_PATH = "/Users/nitishbhattad/Desktop/kafka-fraud-detection/data/paysim.csv"

print("📄 Loading PaySim dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Keep only needed columns + ground-truth label
required_cols = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"❌ Missing columns in PaySim CSV: {missing}")

df = df[required_cols].copy()

# Optional: downsample for speed (you can increase later)
# This keeps the overall structure but makes streaming faster
if len(df) > 500_000:
    df = df.sample(500_000, random_state=42).reset_index(drop=True)

fraud_df = df[df["isFraud"] == 1].reset_index(drop=True)
legit_df = df[df["isFraud"] == 0].reset_index(drop=True)

print(f"✅ Loaded {len(df)} rows total "
      f"({len(fraud_df)} fraud, {len(legit_df)} legit)")

if fraud_df.empty:
    raise ValueError("❌ No fraud rows found in the dataset (isFraud == 1).")

# ============================================
# 2. Kafka producer setup
# ============================================

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

TOPIC = "transactions"

# Probability that a sent transaction is fraud (oversampling for demo)
P_FRAUD = 0.20  # 20% of streamed events will be true frauds

print(f"🔌 Connected to Kafka. Topic = '{TOPIC}'")
print(f"🎯 Target streaming fraud ratio ~ {int(P_FRAUD * 100)}%")

# ============================================
# 3. Helper functions to pick rows
# ============================================

def sample_fraud_row():
    idx = random.randint(0, len(fraud_df) - 1)
    return fraud_df.iloc[idx]

def sample_legit_row():
    idx = random.randint(0, len(legit_df) - 1)
    return legit_df.iloc[idx]

# ============================================
# 4. Main streaming loop
# ============================================

def main():
    i = 0
    try:
        while True:
            # Decide whether to send fraud or legit
            if random.random() < P_FRAUD:
                row = sample_fraud_row()
            else:
                row = sample_legit_row()

            record = {
                "transaction_id": f"txn_{i}",
                "timestamp": str(time.time()),
                "type": row["type"],
                "amount": float(row["amount"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "newbalanceOrig": float(row["newbalanceOrig"]),
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "newbalanceDest": float(row["newbalanceDest"]),
                # Ground truth label from PaySim
                "is_fraud_true": int(row["isFraud"]),
            }

            producer.send(TOPIC, record)
            print("📤 Sent:", record)

            i += 1
            # Control streaming speed: 5 events/sec
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n🛑 Stopping producer (KeyboardInterrupt)")
    finally:
        producer.flush()
        producer.close()
        print("✅ Producer closed cleanly.")


if __name__ == "__main__":
    main()
