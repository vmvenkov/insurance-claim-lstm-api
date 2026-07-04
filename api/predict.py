import os
import torch
import torch.nn.functional as F

from trainer.model import ClaimLSTM
from trainer.utils import encode_text


MODEL_PATH = "model/claim_lstm.pt"


class ClaimPredictor:
    def __init__(self):
        self.model = None
        self.checkpoint = None

    def load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Please train the model first.")

        self.checkpoint = torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )

        self.model = ClaimLSTM(
            vocab_size=len(self.checkpoint["vocab"]),
            insurance_classes=len(self.checkpoint["insurance_classes"]),
            claim_classes=len(self.checkpoint["claim_classes"]),
            severity_classes=len(self.checkpoint["severity_classes"])
        )

        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, description):
        if self.model is None:
            self.load()

        vocab = self.checkpoint["vocab"]
        max_len = self.checkpoint["max_len"]

        encoded = encode_text(description, vocab, max_len)
        x = torch.tensor([encoded], dtype=torch.long)

        with torch.no_grad():
            insurance_logits, claim_logits, severity_logits = self.model(x)

            insurance_probs = F.softmax(insurance_logits, dim=1)
            claim_probs = F.softmax(claim_logits, dim=1)
            severity_probs = F.softmax(severity_logits, dim=1)

            insurance_conf, insurance_idx = torch.max(insurance_probs, dim=1)
            claim_conf, claim_idx = torch.max(claim_probs, dim=1)
            severity_conf, severity_idx = torch.max(severity_probs, dim=1)

        insurance_type = self.checkpoint["insurance_classes"][insurance_idx.item()]
        claim_type = self.checkpoint["claim_classes"][claim_idx.item()]
        severity = self.checkpoint["severity_classes"][severity_idx.item()]

        return {
            "insurance_type": insurance_type,
            "insurance_confidence": round(insurance_conf.item(), 4),
            "claim_type": claim_type,
            "claim_confidence": round(claim_conf.item(), 4),
            "severity": severity,
            "severity_confidence": round(severity_conf.item(), 4),
            "recommendation": get_recommendation(
                insurance_type,
                claim_type,
                severity
            )
        }


def get_recommendation(insurance_type, claim_type, severity):
    if severity == "High":
        return "Необходима е спешна проверка, оглед и детайлна оценка на щетата."

    if claim_type in ["Theft", "Fire", "Flood"]:
        return "Необходимо е представяне на документи, снимки и официален протокол."

    if insurance_type == "Travel":
        return "Необходимо е представяне на документи от превозвач или медицинско заведение."

    return "Необходим е стандартен оглед и проверка на покритието по полицата."
