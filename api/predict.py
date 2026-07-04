import os
import re
import torch
import torch.nn as nn


MODEL_PATH = "model/claim_lstm.pt"
EMBEDDING_DIM = 64
HIDDEN_DIM = 128


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9\s-]", "", text)
    return text.split()


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


def encode_text(text, vocab, max_len):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please train the model first.")

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

    model = ClaimLSTM(
        vocab_size=len(checkpoint["vocab"]),
        insurance_classes=len(checkpoint["insurance_classes"]),
        claim_classes=len(checkpoint["claim_classes"]),
        severity_classes=len(checkpoint["severity_classes"])
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def predict_claim(description):
    model, checkpoint = load_model()

    vocab = checkpoint["vocab"]
    max_len = checkpoint["max_len"]

    encoded = encode_text(description, vocab, max_len)
    x = torch.tensor([encoded], dtype=torch.long)

    with torch.no_grad():
        insurance_pred, claim_pred, severity_pred = model(x)

        insurance_idx = torch.argmax(insurance_pred, dim=1).item()
        claim_idx = torch.argmax(claim_pred, dim=1).item()
        severity_idx = torch.argmax(severity_pred, dim=1).item()

    insurance_type = checkpoint["insurance_classes"][insurance_idx]
    claim_type = checkpoint["claim_classes"][claim_idx]
    severity = checkpoint["severity_classes"][severity_idx]

    recommendation = get_recommendation(insurance_type, claim_type, severity)

    return {
        "description": description,
        "insurance_type": insurance_type,
        "claim_type": claim_type,
        "severity": severity,
        "recommendation": recommendation
    }


def get_recommendation(insurance_type, claim_type, severity):
    if severity == "High":
        return "Необходима е спешна проверка, оглед и детайлна оценка на щетата."
    elif claim_type in ["Theft", "Fire", "Flood"]:
        return "Необходимо е представяне на документи, снимки и официален протокол."
    elif insurance_type == "Travel":
        return "Необходимо е представяне на документи от превозвач или медицинско заведение."
    else:
        return "Необходим е стандартен оглед и проверка на покритието по полицата."
