def build_lstm_net(input_size: int, hidden_size: int, num_layers: int, dropout: float):
    import torch.nn as nn

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out).squeeze(-1)

    return LSTMNet()


def build_transformer_net(input_size: int, d_model: int, nhead: int, num_layers: int, dropout: float):
    import math
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, size: int, max_len: int = 5000):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, size, 2) * (-math.log(10000.0) / size))
            pe = torch.zeros(max_len, size)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1), :]

    class TransformerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            self.dropout = nn.Dropout(dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                dim_feedforward=d_model * 4,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            x = self.dropout(x)
            x = self.encoder(x)
            x = x[:, -1, :]
            return self.fc(x).squeeze(-1)

    return TransformerNet()
