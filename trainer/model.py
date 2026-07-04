import torch.nn as nn


EMBEDDING_DIM = 64
HIDDEN_DIM = 128


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
