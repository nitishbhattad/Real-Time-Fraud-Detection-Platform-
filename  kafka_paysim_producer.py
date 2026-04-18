# ~/Desktop/kafka-fraud-detection/kafka_paysim_producer.py

import json
import time
import pandas as pd
from kafka import KafkaProducer

# ================================
# 1. Load PaySim dataset
# ================================
DATA_PATH = "/Users/nitishbhattad/Desktop/kafka-fraud-detection/data/paysim.csv"

print("📄 Loading PaySim dataset...")
df = pd.read_csv(DATA_PATH)

# Reduce dataset for faster streaming (optional)
df = df.sample(50000, random_state=42).reset_index(drop=True)

# Ensure required columns exist
required_cols = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

# ================================
# 2. Create Kafka Producer
# ================================

print("🔌 Connecting to Kafka Producer...")
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print("🚀 Kafka Producer ready. Starting stream...")

# ================================
# 3. Stream rows to Kafka
# ================================
for i, row in df.iterrows():

    record = {
        "transaction_id": f"txn_{i}",
        "timestamp": str(time.time()),

        "type": row["type"],
        "amount": float(row["amount"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
    }

    producer.send("transactions", record)

    print(f"📤 Sent: {record}")

    time.sleep(0.2)   # send 5 transactions per second

print("✅ Streaming completed!")
producer.flush()
