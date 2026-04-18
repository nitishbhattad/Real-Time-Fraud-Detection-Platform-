# 🛡️ Real-Time Fraud Detection Platform

**End-to-End Streaming ML System with GenAI Explanations**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.4-orange.svg)](https://kafka.apache.org)
[![Spark](https://img.shields.io/badge/Apache%20Spark-3.4-red.svg)](https://spark.apache.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Master's Flagship Project — DSC 550 | University of Massachusetts Dartmouth**  
> Nitish Bhattad 

---

## 📌 Overview

A **production-grade, real-time fraud detection system** that streams financial transactions through a complete ML pipeline — from ingestion to detection to AI-powered explanation — in under 5 seconds per transaction.

Unlike typical fraud detection notebooks, this system is built like a real fintech product:
- **Streaming pipeline** (not batch processing)
- **REST API microservice** (not just a script)
- **Database-backed storage** (not CSV files)
- **GenAI explanations** (not just raw model scores)
- **Containerized deployment** (Docker Compose)

---

## 🏗️ Architecture

```
PaySim Dataset
      │
      ▼
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│   Kafka     │────▶│    Spark    │────▶│  SQLite / Postgres│
│  Producer   │     │  Streaming  │     │    (Alerts DB)    │
│ (5 txn/sec) │     │  Consumer   │     └────────┬─────────┘
└─────────────┘     │             │              │
                    │ ┌─────────┐ │              ▼
                    │ │   ML    │ │     ┌─────────────────┐
                    │ │ Model   │ │     │    FastAPI       │
                    │ │  +Rules │ │     │  REST Endpoint  │
                    │ └─────────┘ │     │  /score         │
                    └─────────────┘     │  /score/batch   │
                                        │  /explain       │
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────┐
                                    │  Streamlit Dashboard │
                                    │  ┌───────────────┐  │
                                    │  │  Overview     │  │
                                    │  │  GenAI Analyst│  │
                                    │  │  Live Perf.   │  │
                                    │  │  Search       │  │
                                    │  └───────────────┘  │
                                    │     + OpenAI GPT    │
                                    └─────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔴 **Real-Time Streaming** | Kafka → Spark Structured Streaming pipeline processing 5 transactions/second |
| 🤖 **Hybrid ML Detection** | RandomForest (ROC-AUC 0.9989) + rule-based engine for composite fraud scoring |
| 🧠 **GenAI Explanations** | GPT-4o-mini generates natural-language analyst reports per flagged transaction |
| 📊 **SHAP Explainability** | Feature-level contribution analysis for every fraud prediction |
| 🗄️ **Database-Backed** | SQLAlchemy ORM with SQLite (dev) / PostgreSQL (production) |
| ⚡ **REST API** | FastAPI microservice with `/score`, `/score/batch`, `/explain` endpoints |
| 📈 **Live Performance** | Real-time precision/recall/F1 tracking against ground truth labels |
| 🐳 **Dockerized** | Full stack runs with a single `docker-compose up` command |
| 🚨 **Email Alerts** | Automated HTML email notifications for critical fraud detections |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9989** |
| Precision (Fraud) | **0.70** |
| Recall (Fraud) | **0.81** |
| F1 Score (Fraud) | **0.75** |
| Average Precision (PR-AUC) | **0.8758** |
| False Positive Rate | **0.05%** |
| Threshold | **0.9** (tuned via F1 scan) |
| Training Data | 200,000 PaySim transactions |

The model was trained with `RandomUnderSampler` to handle the heavily imbalanced dataset (0.13% fraud rate), and the decision threshold was selected via a precision-recall-F1 scan rather than defaulting to 0.5.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Message Broker** | Apache Kafka 3.4 |
| **Stream Processing** | Apache Spark 3.4 (Structured Streaming + Pandas UDF) |
| **ML Model** | Scikit-learn RandomForestClassifier (200 estimators) |
| **Explainability** | SHAP KernelExplainer |
| **GenAI** | OpenAI GPT-4o-mini |
| **API** | FastAPI + Uvicorn |
| **Database** | SQLAlchemy ORM → SQLite / PostgreSQL |
| **Dashboard** | Streamlit |
| **Containerization** | Docker + Docker Compose |
| **Data** | PaySim (6.3M synthetic financial transactions) |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/nitishbhattad/fraud-detection-platform.git
cd fraud-detection-platform

# Set your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Launch everything
docker-compose up --build
```

Open **http://localhost:8501** for the dashboard.  
Open **http://localhost:8000/docs** for the API.

### Option 2: Local Setup

**Prerequisites:** Python 3.11+, Java 17, Apache Kafka installed at `~/kafka`

```bash
# Create virtual environment
python3 -m venv venv311
source venv311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database and migrate existing alerts
python database.py migrate
```

**Start the pipeline (5 terminals):**

```bash
# Terminal 1 — ZooKeeper
cd ~/kafka && bin/zookeeper-server-start.sh config/zookeeper.properties

# Terminal 2 — Kafka Broker
cd ~/kafka && bin/kafka-server-start.sh config/server.properties

# Terminal 3 — Spark Consumer
source venv311/bin/activate
python spark_streaming_consumer_db.py

# Terminal 4 — Kafka Producer
source venv311/bin/activate
python kafka_paysim_producer.py

# Terminal 5 — Dashboard
source venv311/bin/activate
export OPENAI_API_KEY="sk-your-key"
streamlit run dashboard_pro.py
```

---

## 📡 REST API

The FastAPI microservice runs at `http://localhost:8000`. Full docs at `/docs`.

### Score a single transaction

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_001",
    "type": "CASH_OUT",
    "amount": 450000,
    "oldbalanceOrg": 450000,
    "newbalanceOrig": 0,
    "oldbalanceDest": 0,
    "newbalanceDest": 450000
  }'
```

**Response:**
```json
{
  "transaction_id": "txn_001",
  "fraud_probability": 0.91,
  "is_fraud": true,
  "threshold_used": 0.9,
  "risk_level": "CRITICAL",
  "inference_time_ms": 101.42,
  "top_features": {
    "oldbalanceOrg": 0.0741,
    "newbalanceDest": 0.0373,
    "amount": 0.0346
  }
}
```

### Batch scoring

```bash
curl -X POST http://localhost:8000/score/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...]}'
```

### Health check

```bash
curl http://localhost:8000/health
```

---

## 🗂️ Project Structure

```
fraud-detection-platform/
│
├── kafka_paysim_producer.py         # Streams PaySim transactions to Kafka
├── spark_streaming_consumer_db.py   # Spark ML scoring → database
├── spark_streaming_consumer.py      # Spark ML scoring → CSV (legacy)
│
├── api_server.py                    # FastAPI REST microservice
├── database.py                      # SQLAlchemy models + CRUD operations
│
├── dashboard_pro.py                 # Production Streamlit dashboard (4 tabs)
├── dashboard_genai.py               # Streamlit dashboard (simplified)
│
├── model_evaluation.py              # Comprehensive model evaluation script
├── email_alerts.py                  # Automated email alert monitor
│
├── docker-compose.yml               # Full stack Docker orchestration
├── Dockerfile.producer              # Kafka producer container
├── Dockerfile.spark                 # Spark consumer container
├── Dockerfile.dashboard             # Streamlit dashboard container
│
├── models/
│   └── fraud_model_paysim_rf.pkl   # Trained RandomForest pipeline
│
├── data/
│   └── paysim.csv                  # PaySim dataset (6.3M rows)
│
└── requirements.txt
```

---

## 📈 Model Evaluation

Run the full evaluation suite:

```bash
python model_evaluation.py
```

Generates 7 publication-quality plots:
- Confusion matrix (raw + normalized)
- ROC curve with operating point
- Precision-Recall curve
- Threshold analysis (P/R/F1 vs threshold)
- Feature importance (RandomForest Gini)
- Score distribution (fraud vs legit)
- Detection strategy comparison (ML vs Rules vs Hybrid)

---

## 🗃️ Database

The system uses SQLite by default (zero setup). Switch to PostgreSQL with one environment variable:

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/fraud_db"
```

Migrate existing CSV alert files to the database:

```bash
python database.py migrate
```

---

## 🔔 Email Alerts

Configure automated email alerts for critical fraud detections:

```bash
export ALERT_EMAIL_FROM="your@gmail.com"
export ALERT_EMAIL_PASSWORD="your-app-password"  # Gmail App Password
export ALERT_EMAIL_TO="analyst@company.com"

# Test first
python email_alerts.py test

# Run live monitor
python email_alerts.py
```

---

## 📋 Requirements

```
kafka-python
pandas
scikit-learn==1.4.2
joblib
numpy
pyspark==3.4.1
pyarrow
streamlit
altair
shap
openai
fastapi
uvicorn
sqlalchemy
setuptools
streamlit-autorefresh
jinja2
```

---

## 🎯 Detection Strategy

The system uses a **hybrid approach** combining ML and rule-based detection:

```
Transaction Input
       │
       ├──▶ RandomForest Classifier ──▶ ML Score (0-1)
       │         (200 estimators)
       │
       ├──▶ Rule 1: Amount > $400,000 ──▶ high_amount_flag
       │
       └──▶ Rule 2: Zero balance drain ──▶ zero_balance_flag
                (TRANSFER/CASH_OUT with
                 oldBalance → 0)

Final Decision = ML_flag OR rule_flag

Severity:
  CRITICAL → ML score ≥ 0.9 OR high amount flag
  HIGH     → ML score ≥ 0.6 OR rule flag
  MEDIUM   → ML score ≥ 0.4
  LOW      → ML score < 0.4 but flagged
```

---

## 📚 Dataset

**PaySim** — Synthetic financial transaction dataset simulating mobile money transfers.

- 6,362,620 total transactions
- 0.13% fraud rate (heavily imbalanced)
- Features: transaction type, amount, sender/receiver balances
- Ground truth labels for evaluation

---

## 👥 Authors

**Nitish Bhattad** — [LinkedIn](https://linkedin.com/in/nitish-bhattad-457820150) | [GitHub](https://github.com/nitishbhattad)  
**Sowmika Dinesh**

Master's in Data Science — University of Massachusetts Dartmouth  
DSC 550 Flagship Project | December 2025

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
