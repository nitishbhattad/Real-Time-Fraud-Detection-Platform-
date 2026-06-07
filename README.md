# 🛡️ Real-Time Fraud Detection Platform

**End-to-End Streaming ML System with GenAI Explanations**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.4-orange.svg)](https://kafka.apache.org)
[![Spark](https://img.shields.io/badge/Apache%20Spark-3.4-red.svg)](https://spark.apache.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Master's Flagship Project — DSC 550 | University of Massachusetts Dartmouth**
> **Nitish Bhattad**

---

## 📌 Overview

A **production-grade, real-time fraud detection system** that streams financial transactions through a complete ML pipeline — from ingestion to detection to AI-powered explanation — in under 5 seconds per transaction.

Unlike typical fraud detection notebooks, this system is built like a real fintech product:
- **Streaming pipeline** (not batch processing)
- **REST API microservice** (not just a script)
- **Database-backed storage** (not CSV files)
- **GenAI explanations** (not just raw model scores)
- **Containerized deployment** (Docker Compose included)

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
└─────────────┘     │  ML + Rules │              │
                    └─────────────┘              ▼
                                        ┌─────────────────┐
                                        │    FastAPI       │
                                        │  /score          │
                                        │  /score/batch    │
                                        │  /explain        │
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────┐
                                    │  Streamlit Dashboard │
                                    │  Overview            │
                                    │  GenAI Analyst       │
                                    │  Live Performance    │
                                    │  Search              │
                                    └─────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔴 **Real-Time Streaming** | Kafka → Spark Structured Streaming processing 5 transactions/second |
| 🤖 **Hybrid ML Detection** | RandomForest (ROC-AUC 0.9989) + rule-based engine |
| 🧠 **GenAI Explanations** | GPT-4o-mini generates natural-language analyst reports per transaction |
| 📊 **SHAP Explainability** | Feature-level contribution analysis for every fraud prediction |
| 🗄️ **Database-Backed** | SQLAlchemy ORM with SQLite (dev) / PostgreSQL (production) |
| ⚡ **REST API** | FastAPI microservice with `/score`, `/score/batch`, `/explain` endpoints |
| 📈 **Live Performance** | Real-time precision/recall/F1 tracking against ground truth labels |
| 🐳 **Docker Support** | Docker Compose files included for containerized deployment |

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
| **Data** | PaySim (6.3M synthetic financial transactions) |

---

## 🚀 Local Setup

**Prerequisites:** Python 3.11+, Java 17, Apache Kafka installed at `~/kafka`

**Step 1: Create virtual environment**
```bash
cd kafka-fraud-detection
python3 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
pip install setuptools
```

**Step 2: Initialize database**
```bash
python database.py
python database.py migrate   # import any existing CSV alerts
```

**Step 3: Start the pipeline (5 terminals)**

```bash
# Terminal 1 — ZooKeeper
cd ~/kafka && bin/zookeeper-server-start.sh config/zookeeper.properties

# Terminal 2 — Kafka Broker
cd ~/kafka && bin/kafka-server-start.sh config/server.properties

# Terminal 3 — Spark Consumer
source venv311/bin/activate && pip install setuptools
python spark_streaming_consumer_db.py

# Terminal 4 — Kafka Producer
source venv311/bin/activate
python kafka_paysim_producer.py

# Terminal 5 — Dashboard
source venv311/bin/activate
export OPENAI_API_KEY="sk-your-key"
streamlit run dashboard_pro.py
```

**Step 6: FastAPI endpoint (optional)**
```bash
source venv311/bin/activate
uvicorn api_server:app --port 8000
```

Open **http://localhost:8501** for the dashboard.
Open **http://localhost:8000/docs** for the API Swagger UI.

---

## 📡 REST API

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

---

## 🗂️ Project Structure

```
fraud-detection-platform/
├── kafka_paysim_producer.py         # Streams PaySim transactions to Kafka
├── spark_streaming_consumer_db.py   # Spark ML scoring → database
├── spark_streaming_consumer.py      # Spark ML scoring → CSV (legacy)
├── api_server.py                    # FastAPI REST microservice
├── database.py                      # SQLAlchemy models + CRUD
├── dashboard_pro.py                 # Production Streamlit dashboard (4 tabs)
├── dashboard_genai.py               # Streamlit dashboard (simplified)
├── model_evaluation.py              # Full evaluation script (7 plots)
├── docker-compose.yml               # Docker Compose orchestration
├── Dockerfile.producer              # Kafka producer container
├── Dockerfile.spark                 # Spark consumer container
├── Dockerfile.dashboard             # Streamlit dashboard container
├── models/                          # Trained model (not tracked in git)
├── data/                            # PaySim dataset (not tracked in git)
└── requirements.txt
```

---

## 📈 Model Evaluation

```bash
python model_evaluation.py
```

Generates 7 PNG plots: confusion matrix, ROC curve, precision-recall curve,
threshold analysis, feature importance, score distribution, and detection
strategy comparison (ML vs Rules vs Hybrid).

---

## 🎯 Detection Strategy

```
Transaction Input
       ├──▶ RandomForest Classifier ──▶ ML Score (0-1)
       ├──▶ Rule 1: Amount > $400,000
       └──▶ Rule 2: Zero balance drain (TRANSFER/CASH_OUT)

Final Decision = ML_flag OR rule_flag

Severity: CRITICAL / HIGH / MEDIUM / LOW
```

---

## 📚 Dataset

**PaySim** — 6,362,620 synthetic financial transactions, 0.13% fraud rate.
Features: type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest.

---

## 👥 Authors

**Nitish Bhattad** — [LinkedIn](https://linkedin.com/in/nitish-bhattad-457820150) | [GitHub](https://github.com/nitishbhattad)


Master's in Data Science — University of Massachusetts Dartmouth | DSC 550

---

## 📄 License

MIT License