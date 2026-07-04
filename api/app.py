import os
import json
import requests
from fastapi import FastAPI

from api.schemas import ClaimRequest, PredictionResponse
from api.predict import ClaimPredictor


app = FastAPI(
    title="Insurance Claim LSTM API",
    description="REST API for analyzing insurance claim descriptions using an LSTM neural network.",
    version="1.0"
)

TRAINER_URL = "http://trainer:8001/train"
INFO_PATH = "model/model_info.json"

predictor = ClaimPredictor()


@app.get("/")
def home():
    return {"message": "Insurance Claim LSTM API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ClaimRequest):
    return predictor.predict(request.description)


@app.post("/train")
def train_model():
    response = requests.post(TRAINER_URL)
    return response.json()


@app.get("/model-info")
def model_info():
    if not os.path.exists(INFO_PATH):
        return {"message": "Model information not found. Please train the model first."}

    with open(INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
