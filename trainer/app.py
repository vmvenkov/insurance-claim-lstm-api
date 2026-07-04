from fastapi import FastAPI
from train import train


app = FastAPI(
    title="Insurance Claim LSTM Trainer",
    description="Training service for the LSTM insurance claim model.",
    version="1.0"
)


@app.get("/")
def home():
    return {
        "message": "Training service is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/train")
def train_model():
    train()
    return {
        "message": "Model training completed successfully."
    }
