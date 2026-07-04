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

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5


def calculate_accuracy(predictions, labels):
    predicted_classes = torch.argmax(predictions, dim=1)
    correct = (predicted_classes == labels).sum().item()
    return correct, labels.size(0)


def evaluate(model, data_loader, criterion):
    model.eval()

    total_loss = 0

    correct_insurance = 0
    correct_claim = 0
    correct_severity = 0
    correct_department = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"]
            insurance_y = batch["insurance"]
            claim_y = batch["claim"]
            severity_y = batch["severity"]
            department_y = batch["department"]

            (
                insurance_pred,
                claim_pred,
                severity_pred,
                department_pred
            ) = model(x)

            loss = (
                criterion(insurance_pred, insurance_y)
                + criterion(claim_pred, claim_y)
                + criterion(severity_pred, severity_y)
                + criterion(department_pred, department_y)
            )

            total_loss += loss.item()

            c, n = calculate_accuracy(insurance_pred, insurance_y)
            correct_insurance += c

            c, _ = calculate_accuracy(claim_pred, claim_y)
            correct_claim += c

            c, _ = calculate_accuracy(severity_pred, severity_y)
            correct_severity += c

            c, _ = calculate_accuracy(department_pred, department_y)
            correct_department += c

            total_samples += n

    return {
        "loss": total_loss / len(data_loader),
        "insurance_accuracy": correct_insurance / total_samples,
        "claim_accuracy": correct_claim / total_samples,
        "severity_accuracy": correct_severity / total_samples,
        "department_accuracy": correct_department / total_samples
    }


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    texts = df["description"].astype(str).tolist()

    vocab = build_vocab(texts)

    insurance_encoder = LabelEncoder()
    claim_encoder = LabelEncoder()
    severity_encoder = LabelEncoder()
    department_encoder = LabelEncoder()

    y_insurance = insurance_encoder.fit_transform(df["insurance_type"])
    y_claim = claim_encoder.fit_transform(df["claim_type"])
    y_severity = severity_encoder.fit_transform(df["severity"])
    y_department = department_encoder.fit_transform(df["department"])

    split_data = train_test_split(
        texts,
        y_insurance,
        y_claim,
        y_severity,
        y_department,
        test_size=0.2,
        random_state=42
    )

    (
        X_train,
        X_val,
        yi_train,
        yi_val,
        yc_train,
        yc_val,
        ys_train,
        ys_val,
        yd_train,
        yd_val
    ) = split_data

    train_dataset = ClaimsDataset(
        X_train,
        yi_train,
        yc_train,
        ys_train,
        yd_train,
        vocab,
        MAX_LEN
    )

    val_dataset = ClaimsDataset(
        X_val,
        yi_val,
        yc_val,
        ys_val,
        yd_val,
        vocab,
        MAX_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = ClaimLSTM(
        vocab_size=len(vocab),
        insurance_classes=len(insurance_encoder.classes_),
        claim_classes=len(claim_encoder.classes_),
        severity_classes=len(severity_encoder.classes_),
        department_classes=len(department_encoder.classes_)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_metrics = {}

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            x = batch["x"]
            insurance_y = batch["insurance"]
            claim_y = batch["claim"]
            severity_y = batch["severity"]
            department_y = batch["department"]

            optimizer.zero_grad()

            (
                insurance_pred,
                claim_pred,
                severity_pred,
                department_pred
            ) = model(x)

            loss = (
                criterion(insurance_pred, insurance_y)
                + criterion(claim_pred, claim_y)
                + criterion(severity_pred, severity_y)
                + criterion(department_pred, department_y)
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Insurance Acc: {val_metrics['insurance_accuracy']:.4f} | "
            f"Claim Acc: {val_metrics['claim_accuracy']:.4f} | "
            f"Severity Acc: {val_metrics['severity_accuracy']:.4f} | "
            f"Department Acc: {val_metrics['department_accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            best_metrics = val_metrics

            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "insurance_classes": insurance_encoder.classes_.tolist(),
                "claim_classes": claim_encoder.classes_.tolist(),
                "severity_classes": severity_encoder.classes_.tolist(),
                "department_classes": department_encoder.classes_.tolist(),
                "max_len": MAX_LEN
            }, MODEL_PATH)

            print("Best model saved.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping activated.")
            break

    model_info = {
        "model": "Bidirectional LSTM multi-output classifier",
        "task": "Insurance claim description analysis",
        "epochs_configured": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "vocab_size": len(vocab),
        "best_validation_loss": best_val_loss,
        "best_metrics": best_metrics,
        "outputs": [
            "insurance_type",
            "claim_type",
            "severity",
            "department"
        ]
    }

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)

    print("Training completed.")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
