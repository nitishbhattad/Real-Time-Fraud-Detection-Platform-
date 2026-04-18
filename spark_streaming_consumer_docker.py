# spark_streaming_consumer_docker.py — Docker-compatible version

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, pandas_udf, lit, when
from pyspark.sql.types import StructType, StringType, DoubleType
import joblib

# ============================================
# Config from environment
# ============================================

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
MODEL_PATH = "/app/models/fraud_model_paysim_rf.pkl"
ALERTS_DIR = "/app/fraud_output/alerts"
CHECKPOINT_DIR = "/app/checkpoints/alerts_ml"
TOPIC = "transactions"

# Thresholds
THRESHOLD_ML = 0.6
HIGH_AMOUNT_THRESHOLD = 400_000.0
ZERO_PATTERN_MIN_AMOUNT = 50_000.0

# ============================================
# Load ML model
# ============================================

print(f"🔄 Loading ML model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded.")

FEATURES = [
    "type", "amount", "oldbalanceOrg",
    "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
]

# ============================================
# Spark session
# ============================================

spark = (
    SparkSession.builder
    .appName("FraudStreamingML-Docker")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "1")
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
broadcast_model = spark.sparkContext.broadcast(model)

# ============================================
# Schema
# ============================================

schema = (
    StructType()
    .add("transaction_id", StringType())
    .add("timestamp", StringType())
    .add("type", StringType())
    .add("amount", DoubleType())
    .add("oldbalanceOrg", DoubleType())
    .add("newbalanceOrig", DoubleType())
    .add("oldbalanceDest", DoubleType())
    .add("newbalanceDest", DoubleType())
    .add("is_fraud_true", DoubleType())
)

# ============================================
# Pandas UDF for ML scoring
# ============================================

@pandas_udf("double")
def predict_proba_udf(type_col, amount_col, oldbalanceOrg_col,
                      newbalanceOrig_col, oldbalanceDest_col, newbalanceDest_col):
    import pandas as pd
    X = pd.DataFrame({
        "type": type_col,
        "amount": amount_col,
        "oldbalanceOrg": oldbalanceOrg_col,
        "newbalanceOrig": newbalanceOrig_col,
        "oldbalanceDest": oldbalanceDest_col,
        "newbalanceDest": newbalanceDest_col,
    })
    proba = broadcast_model.value.predict_proba(X)[:, 1]
    return pd.Series(proba)

# ============================================
# Read from Kafka
# ============================================

print(f"📡 Connecting to Kafka at {KAFKA_BOOTSTRAP}...")

raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

events = raw.selectExpr("CAST(value AS STRING) AS json_str")
parsed = events.select(from_json(col("json_str"), schema).alias("data")).select("data.*")

# ============================================
# Apply ML model & rules
# ============================================

scored = parsed.withColumn(
    "ml_score",
    predict_proba_udf(
        col("type"), col("amount"), col("oldbalanceOrg"),
        col("newbalanceOrig"), col("oldbalanceDest"), col("newbalanceDest"),
    ),
)

scored = scored.withColumn("is_fraud_ml", (col("ml_score") >= lit(THRESHOLD_ML)).cast("int"))
scored = scored.withColumn("high_amount_flag", (col("amount") > lit(HIGH_AMOUNT_THRESHOLD)).cast("int"))
scored = scored.withColumn(
    "zero_balance_mismatch",
    (
        col("type").isin("TRANSFER", "CASH_OUT")
        & (col("oldbalanceOrg") == 0.0)
        & (col("newbalanceOrig") == 0.0)
        & (col("amount") > lit(ZERO_PATTERN_MIN_AMOUNT))
    ).cast("int"),
)

scored = scored.withColumn(
    "rule_flag",
    ((col("high_amount_flag") == 1) | (col("zero_balance_mismatch") == 1)).cast("int"),
)

scored = scored.withColumn(
    "anomaly_score",
    col("ml_score") + lit(0.3) * col("high_amount_flag") + lit(0.2) * col("zero_balance_mismatch"),
)

scored = scored.withColumn(
    "is_fraud_final",
   ((col("is_fraud_ml") == 1) | (col("rule_flag") == 1)).cast("int"),
)

scored = scored.withColumn(
    "severity",
    when((col("is_fraud_final") == 1) & ((col("ml_score") >= 0.9) | (col("high_amount_flag") == 1)), lit("critical"))
    .when((col("is_fraud_final") == 1) & ((col("ml_score") >= 0.6) | (col("rule_flag") == 1)), lit("high"))
    .when((col("is_fraud_final") == 1) & (col("ml_score") >= 0.4), lit("medium"))
    .when(col("is_fraud_final") == 1, lit("low"))
    .otherwise(lit("none"))
)

flagged = scored.filter(col("is_fraud_final") == 1)

# ============================================
# Write alerts to CSV
# ============================================

print(f"📝 Writing alerts to: {ALERTS_DIR}")

(
    flagged.select(
        "transaction_id", "timestamp", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "ml_score", "is_fraud_ml", "high_amount_flag", "zero_balance_mismatch",
        "rule_flag", "anomaly_score", "severity", "is_fraud_final", "is_fraud_true",
    )
    .writeStream.format("csv")
    .option("header", "true")
    .option("path", ALERTS_DIR)
    .option("checkpointLocation", CHECKPOINT_DIR)
    .outputMode("append")
    .trigger(processingTime="5 seconds")
    .start()
    .awaitTermination()
)
