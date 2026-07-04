import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from trainer.dataset import ClaimsDataset
from trainer.model import ClaimLSTM
from trainer.utils import build_vocab, MAX_LEN


DATA_PATH = "data/claims.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "claim_lstm.pt")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5


def evaluate(model, data_loader, criterion):
    model.eval()

    total_loss = 0
    all_true = {
        "insurance": [],
        "claim": [],
        "severity": [],
        "department": []
    }
    all_pred = {
        "insurance": [],
        "claim": [],
        "severity": [],
        "department": []
    }

    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"]
            y = {
                "insurance": batch["insurance"],
                "claim": batch["claim"],
                "severity": batch["severity"],
                "department": batch["department"]
            }

            outputs = model(x)

            loss = (
                criterion(outputs[0], y["insurance"])
                + criterion(outputs[1], y["claim"])
                + criterion(outputs[2], y["severity"])
                + criterion(outputs[3], y["department"])
            )

            total_loss += loss.item()

            for i, key in enumerate(all_true.keys()):
                preds = torch.argmax(outputs[i], dim=1)
                all_true[key].extend(y[key].tolist())
                all_pred[key].extend(preds.tolist())

    metrics = {"loss": total_loss / len(data_loader)}

    for key in all_true.keys():
        correct = sum(
            1 for true_value, pred_value in zip(all_true[key], all_pred[key])
            if true_value == pred_value
        )
        metrics[f"{key}_accuracy"] = correct / len(all_true[key])

    return metrics, all_true, all_pred


def build_reports(all_true, all_pred, encoders):
    reports = {}

    for key, encoder in encoders.items():
        labels = list(range(len(encoder.classes_)))

        reports[key] = {
            "classification_report": classification_report(
                all_true[key],
                all_pred[key],
                labels=labels,
                target_names=encoder.classes_,
                output_dict=True,
                zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                all_true[key],
                all_pred[key],
                labels=labels
            ).tolist()
        }

    return reports


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

    X_temp, X_test, yi_temp, yi_test, yc_temp, yc_test, ys_temp, ys_test, yd_temp, yd_test = train_test_split(
        texts,
        y_insurance,
        y_claim,
        y_severity,
        y_department,
        test_size=0.15,
        random_state=42
    )

    X_train, X_val, yi_train, yi_val, yc_train, yc_val, ys_train, ys_val, yd_train, yd_val = train_test_split(
        X_temp,
        yi_temp,
        yc_temp,
        ys_temp,
        yd_temp,
        test_size=0.1765,
        random_state=42
    )

    train_dataset = ClaimsDataset(X_train, yi_train, yc_train, ys_train, yd_train, vocab, MAX_LEN)
    val_dataset = ClaimsDataset(X_val, yi_val, yc_val, ys_val, yd_val, vocab, MAX_LEN)
    test_dataset = ClaimsDataset(X_test, yi_test, yc_test, ys_test, yd_test, vocab, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

            outputs = model(x)

            loss = (
                criterion(outputs[0], insurance_y)
                + criterion(outputs[1], claim_y)
                + criterion(outputs[2], severity_y)
                + criterion(outputs[3], department_y)
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_metrics, _, _ = evaluate(model, val_loader, criterion)

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

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, all_true, all_pred = evaluate(model, test_loader, criterion)

    encoders = {
        "insurance": insurance_encoder,
        "claim": claim_encoder,
        "severity": severity_encoder,
        "department": department_encoder
    }

    reports = build_reports(all_true, all_pred, encoders)

    model_info = {
        "model": "Bidirectional LSTM multi-output classifier",
        "task": "Insurance claim description analysis",
        "epochs_configured": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "vocab_size": len(vocab),
        "best_validation_loss": best_val_loss,
        "best_validation_metrics": best_metrics,
        "test_metrics": test_metrics,
        "outputs": [
            "insurance_type",
            "claim_type",
            "severity",
            "department"
        ]
    }

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=4)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=4)

    print("Training completed.")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    train()
