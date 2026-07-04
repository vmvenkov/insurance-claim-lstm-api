# Insurance Claim Analyzer with Bidirectional LSTM

A machine learning microservice for analyzing insurance claim descriptions written in Bulgarian.

The system trains a Bidirectional LSTM neural network to classify insurance claims and exposes prediction services through a REST API running in Docker containers.

---

# Features

- Bidirectional LSTM neural network
- Multi-output prediction
- REST API using FastAPI
- Dockerized architecture
- Separate Training Service
- Separate Prediction Service
- Automatic model saving
- Early Stopping
- Batch prediction endpoint
- Confidence scores
- Model metadata

---

# Predicted Outputs

For every claim description the model predicts:

- Insurance Type
- Claim Type
- Severity
- Responsible Department

Example:

Input

```
След силна градушка автомобилът има счупено предно стъкло.
```

Output

```json
{
    "insurance_type": "Auto",
    "claim_type": "Hail",
    "severity": "Medium",
    "department": "Motor Claims"
}
```

---

# Project Structure

```
insurance-claim-lstm-api/
│
├── api/
├── trainer/
├── data/
├── model/
├── Dockerfile.api
├── Dockerfile.trainer
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Architecture

```
                Docker Compose
                     │
      ┌──────────────┴──────────────┐
      │                             │
      ▼                             ▼
 Training Service             Prediction API
      │                             │
      └───────────Shared Model──────┘
```

---

# REST Endpoints

## Train Model

```
POST /train
```

Starts model training.

---

## Predict

```
POST /predict
```

Example request

```json
{
    "description":"Автомобилът е ударен отзад на светофар."
}
```

---

## Batch Predict

```
POST /batch-predict
```

Example

```json
{
    "descriptions":[
        "Автомобилът е ударен.",
        "Апартаментът е наводнен."
    ]
}
```

---

## Model Information

```
GET /model-info
```

---

## Health

```
GET /health
```

---

# Technologies

- Python
- PyTorch
- FastAPI
- Docker
- Docker Compose
- Pandas
- Scikit-learn

---

# Dataset

Synthetic insurance claim dataset.

Approximately 5,700 claim descriptions in Bulgarian.

Classes include:

- Auto
- Property
- Travel
- Health
- Accident
- Liability

---

# Machine Learning

Model:

- Bidirectional LSTM

Training:

- Cross Entropy Loss
- Adam Optimizer
- Early Stopping
- Validation Accuracy

---

# Future Improvements

- Attention Mechanism
- Explainable AI
- Transformer-based models
- Multilingual support
- Web interface
- PostgreSQL integration
