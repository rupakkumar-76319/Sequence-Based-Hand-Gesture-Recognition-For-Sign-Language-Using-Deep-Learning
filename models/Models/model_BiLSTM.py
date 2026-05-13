import torch
import torch.nn as nn

class SignLanguageBiLSTM(nn.Module):
    def __init__(self, input_size=126, hidden_size=256, num_layers=2, num_classes=36):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # 🔥 important
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)