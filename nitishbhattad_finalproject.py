import json
import time
import random
import uuid
from datetime import datetime
from kafka import KafkaProducer

# Kafka configuration
TOPIC = "transactions"
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# PaySim-style transaction types
TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]

def simulate_account_balances(amount):
    """
    Generates PaySim-like balances for sender and receiver
    """
    # Sender old balance is >= amount 90% of time
    old_org = round(random.uniform(amount, amount * 5), 2) if random.random() < 0.9 else 0.0
    new_org = round(old_org - amount, 2)

    # Destination old balance: usually > 0 for CASH_OUT, TRANSFER
    old_dest = round(random.uniform(0, old_org * 2), 2)
    new_dest = round(old_dest + amount, 2)

    return old_org, new_org, old_dest, new_dest


def generate_transaction():
    """
    Generates a PaySim-style transaction event
    """
    txn_type = random.choices(
        TYPES,
        weights=[0.4, 0.25, 0.2, 0.1, 0.05],  # PaySim-like frequency distribution
        k=1
    )[0]

    # Amount distribution similar to PaySim
    amount = round(random.uniform(10, 20000), 2)

    # Generate balances
    old_org, new_org, old_dest, new_dest = simulate_account_balances(amount)

    txn = {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),

        # PaySim features used by ML model
        "type": txn_type,
        "amount": amount,
        "oldbalanceOrg": old_org,
        "newbalanceOrig": new_org,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
    }

    return txn


if __name__ == "__main__":
    print("📡 Starting PaySim-style Kafka producer...")
    while True:
        txn = generate_transaction()
        producer.send(TOPIC, txn)
        print("Sent:", txn)
        time.sleep(0.1)  # ~10 transactions per second
