# Realtime Fraud Detection Pipeline

[![CI](https://github.com/kra0s22/realtime-fraud-detection-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/kra0s22/realtime-fraud-detection-pipeline/actions/workflows/ci.yml)

Event-driven, real-time fraud detection system combining stream processing, an online feature store, and low-latency model inference.

## Architecture

```mermaid
flowchart LR
    P[Transaction Producer] -->|transactions.raw| RP[(Redpanda)]
    RP --> SS[PySpark Structured Streaming]
    RP --> AL[Fraud Alerter]
    SS -->|features| RD[(Redis Feature Store)]
    SS --> DL[(Delta Lake)]
    DL --> TR[Model Training]
    TR --> MLF[MLflow Registry]
    MLF --> API[FastAPI Inference]
    MLF --> AL
    AL -->|transactions.alerts| RP
    AL -->|recent alerts| RD
    API --> RD
    API -->|prediction| CL[Client]
```

## Components

| Component                  | Role                                                              |
|----------------------------|-------------------------------------------------------------------|
| Redpanda                   | Kafka-compatible event broker ingesting raw transaction events     |
| PySpark Structured Streaming | Stream processing, feature computation, Delta Lake sink           |
| Delta Lake                 | Versioned storage of raw + enriched transactions                   |
| Redis                      | Online feature store for low-latency feature lookups               |
| FastAPI                    | Synchronous inference endpoint                                     |
| Fraud Alerter              | Stream-scored alerts for suspicious transactions (event-driven)    |
| MLflow                     | Experiment tracking + model registry                               |
| Docker Compose             | Local single-host orchestration of the full pipeline              |

## Quickstart

```bash
# 1. (Optional) Customize configuration
cp .env.example .env

# 2. Build and start the full pipeline
docker compose up --build

# 3. Follow the stream processing job
docker compose logs -f streaming producer
```

## Train the Model

Train a classifier and register it in MLflow (`fraud-detector` → alias `Production`):

```bash
# Pipeline must be running (for MLflow + Redis + the accumulated Delta table)
docker compose up -d

# Run the one-off training job (profiles: train)
docker compose run --rm train

# Restart the API so it loads the newly registered model
docker compose restart api
```

Training reads the transactions accumulated in **Delta Lake**; it falls back to synthetic data only if the table is unavailable.

## Monitor the Model

Evaluate the Production model against fresh data and log metrics to MLflow:

```bash
# Pipeline must be running (for MLflow)
docker compose up -d

# Run the one-off monitoring job (profiles: monitor)
docker compose run --rm evaluate
```

The `fraud-detector-monitoring` run records accuracy, precision, recall, F1 and ROC-AUC so you can track model quality over time.

## Run Tests (no local Python required)

```bash
# Build the test image once (pyspark + test deps)
docker build -t fraud-pipeline-test -f docker/test/Dockerfile .

# Run the whole suite
docker run --rm -v "${PWD}:/app" -w /app --user root --entrypoint bash fraud-pipeline-test \
  -c "PYTHONPATH=/app/src python -m pytest tests -q --disable-warnings"
```

| Service        | Endpoint                                  |
|----------------|-------------------------------------------|
| FastAPI        | http://localhost:8000                     |
| Spark UI       | http://localhost:8080                     |
| MLflow UI      | http://localhost:5000                     |
| Redpanda Admin | http://localhost:9644                     |
| Redis          | redis://localhost:6379                    |

Stop everything with `docker compose down` (add `-v` to also drop data volumes).

## Repository Layout

```
.
├── docker/                  # Container definitions per service
│   ├── api/                 # FastAPI inference image
│   ├── producer/            # Transaction generator image
│   ├── alerter/             # Fraud alerter image
│   ├── spark/               # Spark image with Delta Lake + Kafka jars
│   ├── train/               # Model training image (MLflow + scikit-learn)
│   └── test/                # Dev image for running the pytest suite
├── src/
│   ├── producer/            # Event producer (Redpanda)
│   ├── streaming/           # PySpark Structured Streaming job
│   ├── features/            # Feature store (Redis) client
│   ├── api/                 # FastAPI inference service
│   ├── ml/                  # Model training / registry (MLflow)
│   └── alerter/             # Realtime fraud alerting (stream scoring)
├── tests/                   # pytest suites (unit + integration)
├── docker-compose.yml       # Local pipeline orchestration
└── pyproject.toml           # Project metadata + pytest config
```
