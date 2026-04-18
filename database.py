# ~/Desktop/kafka-fraud-detection/database.py
# ============================================================
# 🛡️ Database Layer — PostgreSQL Alert Storage
# ============================================================
# Replaces CSV file storage with proper database
# Works with both PostgreSQL (production) and SQLite (demo)
# ============================================================

import os
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Boolean, Text, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------
# CONFIG — Uses SQLite by default (no setup needed)
# Switch to PostgreSQL by setting DATABASE_URL env var
# Example: export DATABASE_URL="postgresql://user:pass@localhost:5432/fraud_db"
# ---------------------------------------------------

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///fraud_alerts.db"  # SQLite fallback — zero setup
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ---------------------------------------------------
# MODELS
# ---------------------------------------------------

class FraudAlert(Base):
    """Stores every fraud alert from the streaming pipeline."""
    __tablename__ = "fraud_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(50), index=True)
    timestamp = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Transaction details
    type = Column(String(20))
    amount = Column(Float)
    oldbalanceOrg = Column(Float)
    newbalanceOrig = Column(Float)
    oldbalanceDest = Column(Float)
    newbalanceDest = Column(Float)

    # ML scoring
    ml_score = Column(Float)
    is_fraud_ml = Column(Integer)

    # Rule-based flags
    high_amount_flag = Column(Integer)
    zero_balance_mismatch = Column(Integer)
    rule_flag = Column(Integer)
    anomaly_score = Column(Float)

    # Final decision
    severity = Column(String(20))
    is_fraud_final = Column(Integer)

    # Ground truth (from producer, if available)
    is_fraud_true = Column(Integer, nullable=True)

    # Indexes for fast queries
    __table_args__ = (
        Index("idx_severity", "severity"),
        Index("idx_timestamp", "timestamp"),
        Index("idx_ml_score", "ml_score"),
    )


class ModelPerformance(Base):
    """Tracks live model performance over time."""
    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)
    window_minutes = Column(Integer)  # rolling window size

    total_predictions = Column(Integer)
    true_positives = Column(Integer)
    false_positives = Column(Integer)
    true_negatives = Column(Integer)
    false_negatives = Column(Integer)

    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    accuracy = Column(Float)


class GenAIExplanation(Base):
    """Caches GenAI explanations to avoid re-generating."""
    __tablename__ = "genai_explanations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(50), unique=True, index=True)
    explanation = Column(Text)
    shap_values = Column(Text)  # JSON string
    generated_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------
# CREATE TABLES
# ---------------------------------------------------

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
    print(f"✅ Database initialized: {DATABASE_URL}")


# ---------------------------------------------------
# CRUD OPERATIONS
# ---------------------------------------------------

def insert_alert(alert_data: dict):
    """Insert a single fraud alert into the database."""
    session = SessionLocal()
    try:
        alert = FraudAlert(**alert_data)
        session.add(alert)
        session.commit()
        return alert.id
    except Exception as e:
        session.rollback()
        print(f"⚠️ DB insert error: {e}")
        return None
    finally:
        session.close()


def insert_alerts_batch(alerts: list[dict]):
    """Insert multiple alerts at once (for Spark batch output)."""
    session = SessionLocal()
    try:
        session.bulk_insert_mappings(FraudAlert, alerts)
        session.commit()
        return len(alerts)
    except Exception as e:
        session.rollback()
        print(f"⚠️ DB batch insert error: {e}")
        return 0
    finally:
        session.close()


def get_recent_alerts(limit: int = 100):
    """Get most recent alerts, newest first."""
    session = SessionLocal()
    try:
        alerts = (
            session.query(FraudAlert)
            .order_by(FraudAlert.timestamp.desc())
            .limit(limit)
            .all()
        )
        return alerts
    finally:
        session.close()


def get_alerts_dataframe(limit: int = 1000):
    """Get alerts as a pandas DataFrame (for dashboard)."""
    import pandas as pd
    session = SessionLocal()
    try:
        alerts = (
            session.query(FraudAlert)
            .order_by(FraudAlert.timestamp.desc())
            .limit(limit)
            .all()
        )
        if not alerts:
            return pd.DataFrame()

        data = [{
            "transaction_id": a.transaction_id,
            "timestamp": a.timestamp,
            "type": a.type,
            "amount": a.amount,
            "oldbalanceOrg": a.oldbalanceOrg,
            "newbalanceOrig": a.newbalanceOrig,
            "oldbalanceDest": a.oldbalanceDest,
            "newbalanceDest": a.newbalanceDest,
            "ml_score": a.ml_score,
            "is_fraud_ml": a.is_fraud_ml,
            "high_amount_flag": a.high_amount_flag,
            "zero_balance_mismatch": a.zero_balance_mismatch,
            "rule_flag": a.rule_flag,
            "anomaly_score": a.anomaly_score,
            "severity": a.severity,
            "is_fraud_final": a.is_fraud_final,
            "is_fraud_true": a.is_fraud_true,
        } for a in alerts]

        return pd.DataFrame(data)
    finally:
        session.close()


def get_live_performance():
    """
    Calculate live model performance by comparing
    predictions vs ground truth.
    """
    import pandas as pd
    session = SessionLocal()
    try:
        # Get all alerts that have ground truth
        alerts = (
            session.query(FraudAlert)
            .filter(FraudAlert.is_fraud_true.isnot(None))
            .all()
        )

        if not alerts:
            return None

        data = [{
            "is_fraud_ml": a.is_fraud_ml,
            "is_fraud_true": a.is_fraud_true,
            "ml_score": a.ml_score,
        } for a in alerts]

        df = pd.DataFrame(data)

        # These are only flagged transactions (is_fraud_final == 1)
        # So we compute metrics on what the ML component predicted
        tp = int(((df["is_fraud_ml"] == 1) & (df["is_fraud_true"] == 1)).sum())
        fp = int(((df["is_fraud_ml"] == 1) & (df["is_fraud_true"] == 0)).sum())
        fn = int(((df["is_fraud_ml"] == 0) & (df["is_fraud_true"] == 1)).sum())
        tn = int(((df["is_fraud_ml"] == 0) & (df["is_fraud_true"] == 0)).sum())

        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0

        return {
            "total_predictions": total,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(accuracy, 4),
        }
    finally:
        session.close()


def cache_explanation(transaction_id: str, explanation: str, shap_json: str = None):
    """Cache a GenAI explanation."""
    session = SessionLocal()
    try:
        existing = session.query(GenAIExplanation).filter_by(transaction_id=transaction_id).first()
        if existing:
            existing.explanation = explanation
            existing.shap_values = shap_json
            existing.generated_at = datetime.utcnow()
        else:
            entry = GenAIExplanation(
                transaction_id=transaction_id,
                explanation=explanation,
                shap_values=shap_json,
            )
            session.add(entry)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"⚠️ Cache error: {e}")
    finally:
        session.close()


def get_cached_explanation(transaction_id: str) -> str:
    """Retrieve a cached explanation if available."""
    session = SessionLocal()
    try:
        entry = session.query(GenAIExplanation).filter_by(transaction_id=transaction_id).first()
        return entry.explanation if entry else None
    finally:
        session.close()


# ---------------------------------------------------
# MIGRATION: Import existing CSV alerts into database
# ---------------------------------------------------

def migrate_csv_to_db(alerts_dir: str):
    """
    One-time migration: load existing CSV/part-* alert files
    into the database.
    """
    import glob
    import pandas as pd

    files = glob.glob(os.path.join(alerts_dir, "*.csv"))
    files += glob.glob(os.path.join(alerts_dir, ".part-*"))
    files += glob.glob(os.path.join(alerts_dir, "part-*"))

    if not files:
        print("No alert files found to migrate.")
        return 0

    total = 0
    for f in files:
        try:
            if os.path.getsize(f) < 10:
                continue
            df = pd.read_csv(f)
            if df.empty:
                continue

            df.columns = [c.split("/")[-1] for c in df.columns]

            records = df.to_dict("records")
            inserted = insert_alerts_batch(records)
            total += inserted
            print(f"  ✓ Migrated {inserted} alerts from {os.path.basename(f)}")
        except Exception as e:
            continue

    print(f"\n✅ Total migrated: {total} alerts")
    return total


# ---------------------------------------------------
# CLI
# ---------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        alerts_dir = "/Users/nitishbhattad/Desktop/kafka-fraud-detection/fraud_output/alerts"
        init_db()
        migrate_csv_to_db(alerts_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "stats":
        init_db()
        perf = get_live_performance()
        if perf:
            print("\n📊 Live Model Performance:")
            for k, v in perf.items():
                print(f"  {k}: {v}")
        else:
            print("No performance data yet.")
    else:
        init_db()
        print("Usage:")
        print("  python database.py          # Initialize database")
        print("  python database.py migrate  # Import CSV alerts into DB")
        print("  python database.py stats    # Show live performance")
