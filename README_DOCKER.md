# 🛡️ Real-Time Fraud Detection Platform — Docker Setup

## One-Command Launch

### Step 1: Copy Docker files to your project
Copy ALL these files into your `kafka-fraud-detection` folder:
```
kafka-fraud-detection/
├── docker-compose.yml
├── Dockerfile.producer
├── Dockerfile.spark
├── Dockerfile.dashboard
├── kafka_paysim_producer_docker.py
├── spark_streaming_consumer_docker.py
├── dashboard_genai_docker.py
├── .env                              ← create from .env.example
├── data/
│   └── paysim.csv                    ← already here
├── models/
│   └── fraud_model_paysim_rf.pkl     ← already here
```

### Step 2: Create your .env file
```bash
cp .env.example .env
```
Then edit `.env` and paste your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-key-here
```

### Step 3: Make sure Docker Desktop is running
Open Docker Desktop app. Wait until it says "Docker is running".

### Step 4: Start everything
```bash
cd ~/Desktop/kafka-fraud-detection
docker-compose up --build
```

First time will take 3-5 minutes (downloading images + building).
After that, it starts in ~30 seconds.

### Step 5: Open the dashboard
Go to: **http://localhost:8501**

Wait 15-30 seconds for alerts to start appearing, then:
1. Your OpenAI key is auto-loaded from .env
2. Select a transaction in the GenAI section
3. Click "Generate AI Explanation"

### Stopping
```bash
docker-compose down
```

### Clean restart (reset all data)
```bash
docker-compose down -v
docker-compose up --build
```

## What's Running

| Service      | Container         | Port  | Description                        |
|-------------|-------------------|-------|------------------------------------|
| ZooKeeper   | fraud-zookeeper   | 2181  | Kafka coordination                 |
| Kafka       | fraud-kafka       | 9092  | Message broker                     |
| Producer    | fraud-producer    | —     | Streams PaySim transactions        |
| Spark       | fraud-spark       | —     | ML scoring + rule engine           |
| Dashboard   | fraud-dashboard   | 8501  | Streamlit + GenAI explanations     |

## Troubleshooting

**"No alerts found yet"**
→ Wait 30 seconds and refresh. Spark needs time to start processing.

**Spark container keeps restarting**
→ Check logs: `docker-compose logs spark-consumer`
→ Usually a scikit-learn version mismatch. The Dockerfiles pin v1.4.2.

**OpenAI key not working**
→ Make sure .env has no spaces: `OPENAI_API_KEY=sk-...` (no quotes needed)
→ Or paste it manually in the sidebar

**View logs for any service**
```bash
docker-compose logs -f producer        # producer logs
docker-compose logs -f spark-consumer  # spark logs  
docker-compose logs -f dashboard       # dashboard logs
```
