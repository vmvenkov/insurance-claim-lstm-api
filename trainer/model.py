import torch
import torch.nn as nn


EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.3


class ClaimLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        insurance_classes,
        claim_classes,
        severity_classes,
        department_classes
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True
        )

        feature_size = HIDDEN_DIM * 2

        self.dropout = nn.Dropout(DROPOUT)

        self.insurance_head = nn.Linear(feature_size, insurance_classes)
        self.claim_head = nn.Linear(feature_size, claim_classes)
        self.severity_head = nn.Linear(feature_size, severity_classes)
        self.department_head = nn.Linear(feature_size, department_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        features = torch.cat((forward_hidden, backward_hidden), dim=1)
        features = self.dropout(features)

        insurance_output = self.insurance_head(features)
        claim_output = self.claim_head(features)
        severity_output = self.severity_head(features)
        department_output = self.department_head(features)

        return (
            insurance_output,
            claim_output,
            severity_output,
            department_output
        )
