# ~/Desktop/kafka-fraud-detection/kafka_producer.py
import json, random, string, time, uuid, datetime
from kafka import KafkaProducer

def rand_id(n=8): return ''.join(random.choices(string.ascii_lowercase, k=n))

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    linger_ms=50,
)

merchants = ["Amazon","Flipkart","Myntra","Nykaa","Zomato"]
devices = ["Mobile","Desktop","Tablet"]
cities = ["West James","Port Ryan","Lake Fred","New Emilybury","Schmidtport","Peggyfurt"]

print("Sending transaction data to Kafka (topic: transactions)... Ctrl+C to stop.")
try:
    while True:
        msg = {
            "transaction_id": str(uuid.uuid4()),
            "user_id": rand_id() + str(random.randint(1,99)),
            "amount": round(random.uniform(50, 10000), 2),
            "location": random.choice(cities),
            "device_type": random.choice(devices),
            "merchant": random.choice(merchants),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "is_fraud": 0
        }
        producer.send("transactions", msg)
        print("Sent:", msg)
        time.sleep(1.0)
except KeyboardInterrupt:
    pass
finally:
    producer.flush()
    producer.close()
