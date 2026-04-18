#!/usr/bin/env bash
# run_all.sh — one-touch demo launcher for your Kafka Fraud Detection project
# Platform: macOS (Apple Silicon OK)
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh start   # start everything
#   ./run_all.sh stop    # stop all services started by this script
#   ./run_all.sh clean   # stop + clear /tmp data (Kafka/ZK logs) and local checkpoints
#   ./run_all.sh status  # show quick port/process status

set -euo pipefail

# --- Paths (edit if your layout changes) ---
KAFKA_DIR="$HOME/kafka"
PROJECT_DIR="$HOME/Desktop/kafka-fraud-detection"
VENV_PY="$PROJECT_DIR/venv311/bin/python"         # your venv Python
SPARK_SUBMIT_BIN="$(command -v spark-submit)"     # expects Spark on PATH
STREAMLIT_BIN="$(command -v streamlit || echo "$VENV_PY -m streamlit")"

# Absolute output/checkpoint paths so Spark & Streamlit agree
OUTPUT_DIR="$PROJECT_DIR/fraud_output"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"

# Logs & PID files
LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="$PROJECT_DIR/.pids"
mkdir -p "$LOG_DIR" "$PID_DIR" "$OUTPUT_DIR"

ZK_PID="$PID_DIR/zk.pid"
KAFKA_PID="$PID_DIR/kafka.pid"
SPARK_PID="$PID_DIR/spark.pid"
# Streamlit runs in foreground so you can see the UI logs; comment if you prefer background.

# --- Config ---
TOPIC="transactions"
BOOTSTRAP="localhost:9092"
ZK_PORT=2181
KAFKA_PORT=9092
STREAMLIT_PORT=8501

# --- Helpers ---
die() { echo "ERROR: $*" >&2; exit 1; }

need() {
  command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"
}

wait_port() {
  local port="$1" name="$2" tries="${3:-60}"
  echo "Waiting for $name on port $port ..."
  for i in $(seq 1 "$tries"); do
    if nc -z localhost "$port" >/dev/null 2>&1; then
      echo "$name is up."
      return 0
    fi
    sleep 1
  done
  die "Timed out waiting for $name on port $port"
}

start_zk() {
  if lsof -i :"$ZK_PORT" >/dev/null 2>&1; then
    echo "ZooKeeper already listening on :$ZK_PORT (leaving it)."
    return 0
  fi
  echo "Starting ZooKeeper..."
  nohup "$KAFKA_DIR/bin/zookeeper-server-start.sh" "$KAFKA_DIR/config/zookeeper.properties" \
    >"$LOG_DIR/zookeeper.log" 2>&1 &
  echo $! > "$ZK_PID"
  wait_port "$ZK_PORT" "ZooKeeper"
}

start_kafka() {
  if lsof -i :"$KAFKA_PORT" >/dev/null 2>&1; then
    echo "Kafka already listening on :$KAFKA_PORT (leaving it)."
  else
    echo "Starting Kafka..."
    nohup "$KAFKA_DIR/bin/kafka-server-start.sh" "$KAFKA_DIR/config/server.properties" \
      >"$LOG_DIR/kafka.log" 2>&1 &
    echo $! > "$KAFKA_PID"
    wait_port "$KAFKA_PORT" "Kafka"
  fi

  # Create topic if missing
  if ! "$KAFKA_DIR/bin/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --list | grep -qx "$TOPIC"; then
    echo "Creating topic '$TOPIC'..."
    "$KAFKA_DIR/bin/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" \
      --create --if-not-exists --topic "$TOPIC" --partitions 1 --replication-factor 1
  else
    echo "Topic '$TOPIC' already exists."
  fi
}

start_spark() {
  if [[ -f "$SPARK_PID" ]] && kill -0 "$(cat "$SPARK_PID")" 2>/dev/null; then
    echo "Spark consumer already running (PID $(cat "$SPARK_PID"))."
    return 0
  fi
  [[ -x "$SPARK_SUBMIT_BIN" ]] || die "spark-submit not found on PATH."

  echo "Starting Spark streaming consumer..."
  # Ensure Python in Spark uses your venv
  export PYSPARK_PYTHON="$VENV_PY"
  export PYSPARK_DRIVER_PYTHON="$VENV_PY"

  nohup "$SPARK_SUBMIT_BIN" \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \
    "$PROJECT_DIR/spark_streaming_consumer.py" \
    >"$LOG_DIR/spark_consumer.log" 2>&1 &
  echo $! > "$SPARK_PID"
  echo "Spark consumer started (PID $(cat "$SPARK_PID"))."
}

start_streamlit() {
  echo "Launching Streamlit UI on http://localhost:${STREAMLIT_PORT} ..."
  cd "$PROJECT_DIR"
  # Streamlit runs in foreground so you can Ctrl+C to end the whole demo.
  # If you prefer background, uncomment nohup line and comment the foreground one.
  # nohup $STREAMLIT_BIN run streamlit_app.py --server.port $STREAMLIT_PORT >"$LOG_DIR/streamlit.log" 2>&1 &
  # echo $! > "$PID_DIR/streamlit.pid"
  exec $STREAMLIT_BIN run "$PROJECT_DIR/streamlit_app.py" --server.port "$STREAMLIT_PORT"
}

stop_pidfile() {
  local file="$1" name="$2"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (PID $pid)..."
      kill "$pid" || true
      sleep 2
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
  fi
}

stop_by_port() {
  local port="$1" name="$2"
  if lsof -i :"$port" -t >/dev/null 2>&1; then
    echo "Stopping $name on port $port ..."
    lsof -i :"$port" -t | xargs -I {} kill {} 2>/dev/null || true
    sleep 2
    lsof -i :"$port" -t | xargs -I {} kill -9 {} 2>/dev/null || true
  fi
}

start_all() {
  need nc
  need lsof

  # Make sure virtualenv python exists (for Streamlit/producer)
  [[ -x "$VENV_PY" ]] || die "Venv Python not found at $VENV_PY (did you create/activate venv311?)."

  start_zk
  start_kafka
  start_spark
  echo
  echo "All backends are up. Opening Streamlit next (this also starts the producer)."
  echo "Logs: $LOG_DIR (zookeeper.log, kafka.log, spark_consumer.log)"
  echo
  start_streamlit  # foreground
}

stop_all() {
  echo "Stopping services started by this script..."
  stop_pidfile "$SPARK_PID" "Spark consumer"
  stop_pidfile "$KAFKA_PID" "Kafka"
  stop_pidfile "$ZK_PID" "ZooKeeper"
  # Also ensure ports are freed (handles cases not started via pidfiles)
  stop_by_port "$STREAMLIT_PORT" "Streamlit"
  stop_by_port "$KAFKA_PORT" "Kafka"
  stop_by_port "$ZK_PORT" "ZooKeeper"
  echo "Stopped."
}

clean_all() {
  stop_all
  echo "Cleaning local state (Kafka/ZK logs, checkpoints, output files)..."
  rm -rf /tmp/kafka-logs /tmp/zookeeper
  rm -rf "$CHECKPOINT_DIR"/*
  # Keep output dir but optional purge alerts to show fresh demo
  rm -rf "$OUTPUT_DIR"/alerts/*
  echo "Clean completed."
}

status_all() {
  echo "----- STATUS -----"
  printf "ZooKeeper (:%-5s): " "$ZK_PORT";   lsof -i :"$ZK_PORT"   | tail -n +2 || true
  printf "Kafka     (:%-5s): " "$KAFKA_PORT"; lsof -i :"$KAFKA_PORT" | tail -n +2 || true
  printf "Streamlit (:%-5s): " "$STREAMLIT_PORT"; lsof -i :"$STREAMLIT_PORT" | tail -n +2 || true
  echo "PIDs dir: $PID_DIR"
  ls -l "$PID_DIR" 2>/dev/null || true
  echo "Logs dir: $LOG_DIR"
  ls -l "$LOG_DIR" 2>/dev/null || true
  echo "------------------"
}

cmd="${1:-help}"
case "$cmd" in
  start)  start_all ;;
  stop)   stop_all ;;
  clean)  clean_all ;;
  status) status_all ;;
  *)
    cat <<EOF
Usage: $0 {start|stop|clean|status}

start  - Start ZooKeeper, Kafka, Spark consumer, then run Streamlit UI (foreground).
stop   - Stop services started by this script and free ports 2181/9092/8501.
clean  - stop + remove /tmp/kafka-logs, /tmp/zookeeper, checkpoints, and old alerts.
status - Show quick port/process and logs/PIDs info.

Tip: If you changed fraud logic, run: $0 clean && $0 start
EOF
    ;;
esac
