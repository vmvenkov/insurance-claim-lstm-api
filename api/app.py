import os
import json
import requests
from fastapi import FastAPI, HTTPException

from api.schemas import ClaimRequest, BatchClaimRequest, PredictionResponse
from api.predict import ClaimPredictor


app = FastAPI(
    title="Insurance Claim LSTM API",
    description="REST API for analyzing insurance claim descriptions using a Bidirectional LSTM neural network.",
    version="1.0"
)

TRAINER_URL = "http://trainer:8001/train"
INFO_PATH = "model/model_info.json"
MODEL_PATH = "model/claim_lstm.pt"

predictor = ClaimPredictor()


@app.get("/")
def home():
    return {"message": "Insurance Claim LSTM API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ClaimRequest):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=404,
            detail="Model not found. Please train the model first using POST /train."
        )

    return predictor.predict(request.description)


@app.post("/batch-predict")
def batch_predict(request: BatchClaimRequest):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=404,
            detail="Model not found. Please train the model first using POST /train."
        )

    results = []

    for description in request.descriptions:
        results.append(
            {
                "description": description,
                "prediction": predictor.predict(description)
            }
        )

    return {"results": results}


@app.post("/train")
def train_model():
    try:
        response = requests.post(TRAINER_URL, timeout=600)
        result = response.json()

        if os.path.exists(MODEL_PATH):
            predictor.reload()

        return {
            "training_result": result,
            "model_reloaded": os.path.exists(MODEL_PATH)
        }

    except requests.exceptions.RequestException as error:
        raise HTTPException(
            status_code=500,
            detail=f"Training service error: {str(error)}"
        )


@app.get("/model-info")
def model_info():
    if not os.path.exists(INFO_PATH):
        return {
            "message": "Model information not found. Please train the model first."
        }

    with open(INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
