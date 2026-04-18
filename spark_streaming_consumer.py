# ~/Desktop/kafka-fraud-detection/spark_streaming_consumer.py

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_json,
    pandas_udf,
    lit,
    when,
)
from pyspark.sql.types import (
    StructType,
    StringType,
    DoubleType,
)

import joblib

# ============================================
# Paths & constants
# ============================================

BASE_DIR = "/Users/nitishbhattad/Desktop/kafka-fraud-detection"

MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model_paysim_rf.pkl")

# Folder where Spark writes alerts (used by Streamlit)
ALERTS_DIR = os.path.join(BASE_DIR, "fraud_output", "alerts")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "alerts_ml")

TOPIC = "transactions"

# ML + rule thresholds
THRESHOLD_ML = 0.6                  # ML probability threshold for fraud
HIGH_AMOUNT_THRESHOLD = 400_000.0   # “very large” transaction
ZERO_PATTERN_MIN_AMOUNT = 50_000.0  # PaySim-style zero-balance fraud

# ============================================
# Load ML model (RandomForest pipeline)
# ============================================

print(f"🔄 Loading ML model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded.")
print("✅ Using ML threshold:", THRESHOLD_ML)

FEATURES = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

print("✅ Model features:", FEATURES)

# ============================================
# Spark session
# ============================================

spark = (
    SparkSession.builder
    .appName("FraudStreamingML")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "1")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1",
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# Broadcast the sklearn model so executors can use it
broadcast_model = spark.sparkContext.broadcast(model)

# ============================================
# Schema for incoming Kafka JSON
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
    # optional: if producer sends ground truth
    .add("is_fraud_true", DoubleType())
)

# ============================================
# Pandas UDF for ML scoring
# ============================================

@pandas_udf("double")
def predict_proba_udf(
    type_col,
    amount_col,
    oldbalanceOrg_col,
    newbalanceOrig_col,
    oldbalanceDest_col,
    newbalanceDest_col,
):
    import pandas as pd

    X = pd.DataFrame(
        {
            "type": type_col,
            "amount": amount_col,
            "oldbalanceOrg": oldbalanceOrg_col,
            "newbalanceOrig": newbalanceOrig_col,
            "oldbalanceDest": oldbalanceDest_col,
            "newbalanceDest": newbalanceDest_col,
        }
    )

    proba = broadcast_model.value.predict_proba(X)[:, 1]
    return pd.Series(proba)


# ============================================
# 1) Read from Kafka
# ============================================

raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "localhost:9092")
    .option("subscribe", TOPIC)
    .option("startingOffsets", "latest")
    .load()
)

events = raw.selectExpr("CAST(value AS STRING) AS json_str")
parsed = events.select(from_json(col("json_str"), schema).alias("data")).select(
    "data.*"
)

# ============================================
# 2) Apply ML model & rules
# ============================================

# ML probability
scored = parsed.withColumn(
    "ml_score",
    predict_proba_udf(
        col("type"),
        col("amount"),
        col("oldbalanceOrg"),
        col("newbalanceOrig"),
        col("oldbalanceDest"),
        col("newbalanceDest"),
    ),
)

# ML binary decision
scored = scored.withColumn(
    "is_fraud_ml",
    (col("ml_score") >= lit(THRESHOLD_ML)).cast("int"),
)

# Rule 1: very high amount
scored = scored.withColumn(
    "high_amount_flag",
    (col("amount") > lit(HIGH_AMOUNT_THRESHOLD)).cast("int"),
)

# Rule 2: suspicious zero pattern (PaySim-style)
scored = scored.withColumn(
    "zero_balance_mismatch",
    (
        col("type").isin("TRANSFER", "CASH_OUT")
        & (col("oldbalanceOrg") == 0.0)
        & (col("newbalanceOrig") == 0.0)
        & (col("amount") > lit(ZERO_PATTERN_MIN_AMOUNT))
    ).cast("int"),
)

# Combined rule flag (convert ints -> booleans then OR)
scored = scored.withColumn(
    "rule_flag",
    (
        (col("high_amount_flag") == 1)
        | (col("zero_balance_mismatch") == 1)
    ).cast("int"),
)

# Anomaly score = ML + rules
scored = scored.withColumn(
    "anomaly_score",
    col("ml_score")
    + lit(0.3) * col("high_amount_flag")
    + lit(0.2) * col("zero_balance_mismatch"),
)

# Final decision: either ML or rules say fraud
scored = scored.withColumn(
    "is_fraud_final",
    (col("is_fraud_ml") | col("rule_flag")).cast("int"),
)

# Severity levels (for dashboard)
scored = scored.withColumn(
    "severity",
    when(
        (col("is_fraud_final") == 1)
        & ((col("ml_score") >= 0.9) | (col("high_amount_flag") == 1)),
        lit("critical"),
    )
    .when(
        (col("is_fraud_final") == 1)
        & ((col("ml_score") >= 0.6) | (col("rule_flag") == 1)),
        lit("high"),
    )
    .when(
        (col("is_fraud_final") == 1) & (col("ml_score") >= 0.4),
        lit("medium"),
    )
    .when(col("is_fraud_final") == 1, lit("low"))
    .otherwise(lit("none"))
)

# Keep only actual alerts for the CSV output
flagged = scored.filter(col("is_fraud_final") == 1)

# ============================================
# 3) Write alerts to CSV for Streamlit
# ============================================

(
    flagged.select(
        "transaction_id",
        "timestamp",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "ml_score",
        "is_fraud_ml",
        "high_amount_flag",
        "zero_balance_mismatch",
        "rule_flag",
        "anomaly_score",
        "severity",
        "is_fraud_final",
        "is_fraud_true",  # if present
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
