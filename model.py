import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepHbNetSegmental(nn.Module):
    def __init__(self, input_dim=2304, segment_len=60, hidden_dim=128):
        super(DeepHbNetSegmental, self).__init__()

        # 1D Conv over 2304-dim features for each segment
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2),  # (B×60×1×2304) → (B×60×8×2304)
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),  # (B×60×16×2304) → (B×60×16×64)
        )

        # Flatten CNN output for LSTM: (B, 60, 16×64)
        self.lstm_input_dim = 16 * 64
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, 60, 2304)
        B, T, F = x.shape

        # Reshape for CNN: (B*T, 1, 2304)
        x = x.view(B * T, 1, F)
        x = self.cnn(x)  # (B*T, 16, 64)
        x = x.view(B, T, -1)  # (B, 60, 1024)

        lstm_out, _ = self.lstm(x)  # (B, 60, 2*hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # Use last time step

        out = self.fc(last_hidden)  # (B, 1)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)

        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out += identity
        return F.relu(out)

class DeepHbNetFlexible(nn.Module):
    def __init__(self, input_size=2304, dropout_rate=0.2, num_blocks=4):
        super(DeepHbNetFlexible, self).__init__()

        # Define layer sizes
        hidden_sizes = [1024, 512, 256, 128]
        if num_blocks < 1 or num_blocks > len(hidden_sizes):
            raise ValueError(f"num_blocks must be between 1 and {len(hidden_sizes)}")

        self.blocks = nn.ModuleList()
        in_features = input_size
        for i in range(num_blocks):
            out_features = hidden_sizes[i]
            self.blocks.append(ResidualBlock(in_features, out_features, dropout_rate))
            in_features = out_features

        self.output = nn.Linear(in_features, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output(x)
