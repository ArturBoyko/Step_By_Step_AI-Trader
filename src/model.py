# src/model.py
import torch
import torch.nn as nn

class Encoder1DCNN_LSTM(nn.Module):
    def __init__(self, in_channels=34, latent_dim=128, cnn_hidden=64, lstm_hidden=128, lstm_layers=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, cnn_hidden, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(cnn_hidden, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(input_size=cnn_hidden, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=False, dropout=0.0)
        self.proj = nn.Linear(lstm_hidden, latent_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (B, C, T)
        x = self.conv(x)                # (B, Hc, T)
        x = x.permute(0, 2, 1)          # (B, T, Hc)
        out, (h_n, c_n) = self.lstm(x)  # h_n: (layers, B, Hl)
        h_last = h_n[-1]                # (B, Hl)
        z = self.proj(h_last)           # (B, latent_dim)
        return z

class PredictionHead(nn.Module):
    def __init__(self, latent_dim=128, hidden=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.out_bin = nn.Linear(hidden, 1)
        self.out_cont = nn.Linear(hidden, 1)

    def forward(self, z):
        h = self.mlp(z)
        logit = self.out_bin(h).squeeze(-1)
        ycont = self.out_cont(h).squeeze(-1)
        return logit, ycont
