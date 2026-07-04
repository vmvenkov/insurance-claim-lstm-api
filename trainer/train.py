import os
import json
import re
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


DATA_PATH = "data/claims.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "claim_lstm.pt")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

MAX_LEN = 30
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 0.001


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9\s-]", "", text)
    return text.split()


def build_vocab(texts):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_text(text, vocab):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) < MAX_LEN:
        ids += [vocab["<PAD>"]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return ids


class ClaimsDataset(Dataset):
    def __init__(self, texts, y_insurance, y_claim, y_severity, vocab):
        self.texts = texts
        self.y_insurance = y_insurance
        self.y_claim = y_claim
        self.y_severity = y_severity
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode_text(self.texts[idx], self.vocab)

        return {
            "x": torch.tensor(x, dtype=torch.long),
            "insurance": torch.tensor(self.y_insurance[idx], dtype=torch.long),
            "claim": torch.tensor(self.y_claim[idx], dtype=torch.long),
            "severity": torch.tensor(self.y_severity[idx], dtype=torch.long),
        }


class ClaimLSTM(nn.Module):
    def __init__(self, vocab_size, insurance_classes, claim_classes, severity_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.insurance_head = nn.Linear(HIDDEN_DIM, insurance_classes)
        self.claim_head = nn.Linear(HIDDEN_DIM, claim_classes)
        self.severity_head = nn.Linear(HIDDEN_DIM, severity_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)

        features = self.dropout(hidden[-1])

        insurance_output = self.insurance_head(features)
        claim_output = self.claim_head(features)
        severity_output = self.severity_head(features)

        return insurance_output, claim_output, severity_output


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

    train_dataset = ClaimsDataset(X_train, yi_train, yc_train, ys_train, vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

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

            loss_insurance = criterion(insurance_pred, insurance_y)
            loss_claim = criterion(claim_pred, claim_y)
            loss_severity = criterion(severity_pred, severity_y)

            loss = loss_insurance + loss_claim + loss_severity

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
