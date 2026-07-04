import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from trainer.dataset import ClaimsDataset
from trainer.model import ClaimLSTM
from trainer.utils import build_vocab, MAX_LEN


DATA_PATH = "data/claims.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "claim_lstm.pt")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    texts = df["description"].astype(str).tolist()

    vocab = build_vocab(texts)

    insurance_encoder = LabelEncoder()
    claim_encoder = LabelEncoder()
    severity_encoder = LabelEncoder()

    y_insurance = insurance_encoder.fit_transform(df["insurance_type"])
    y_claim = claim_encoder.fit_transform(df["claim_type"])
    y_severity = severity_encoder.fit_transform(df["severity"])

    X_train, X_test, yi_train, yi_test, yc_train, yc_test, ys_train, ys_test = train_test_split(
        texts,
        y_insurance,
        y_claim,
        y_severity,
        test_size=0.2,
        random_state=42
    )

    train_dataset = ClaimsDataset(
        X_train,
        yi_train,
        yc_train,
        ys_train,
        vocab,
        MAX_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = ClaimLSTM(
        vocab_size=len(vocab),
        insurance_classes=len(insurance_encoder.classes_),
        claim_classes=len(claim_encoder.classes_),
        severity_classes=len(severity_encoder.classes_)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch in train_loader:
            x = batch["x"]
            insurance_y = batch["insurance"]
            claim_y = batch["claim"]
            severity_y = batch["severity"]

            optimizer.zero_grad()

            insurance_pred, claim_pred, severity_pred = model(x)

            loss = (
                criterion(insurance_pred, insurance_y)
                + criterion(claim_pred, claim_y)
                + criterion(severity_pred, severity_y)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "insurance_classes": insurance_encoder.classes_.tolist(),
        "claim_classes": claim_encoder.classes_.tolist(),
        "severity_classes": severity_encoder.classes_.tolist(),
        "max_len": MAX_LEN
    }, MODEL_PATH)

    model_info = {
        "model": "LSTM multi-output classifier",
        "task": "Insurance claim description analysis",
        "epochs": EPOCHS,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "vocab_size": len(vocab),
        "outputs": [
            "insurance_type",
            "claim_type",
            "severity"
        ]
    }

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)

    print("Training completed.")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
