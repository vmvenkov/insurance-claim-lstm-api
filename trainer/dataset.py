import torch
from torch.utils.data import Dataset

from trainer.utils import encode_text


class ClaimsDataset(Dataset):
    def __init__(
        self,
        texts,
        insurance_labels,
        claim_labels,
        severity_labels,
        vocab,
        max_len
    ):
        self.texts = texts
        self.insurance_labels = insurance_labels
        self.claim_labels = claim_labels
        self.severity_labels = severity_labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = encode_text(
            self.texts[idx],
            self.vocab,
            self.max_len
        )

        return {
            "x": torch.tensor(encoded, dtype=torch.long),
            "insurance": torch.tensor(
                self.insurance_labels[idx],
                dtype=torch.long
            ),
            "claim": torch.tensor(
                self.claim_labels[idx],
                dtype=torch.long
            ),
            "severity": torch.tensor(
                self.severity_labels[idx],
                dtype=torch.long
            ),
        }
