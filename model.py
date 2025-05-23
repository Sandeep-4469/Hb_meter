import torch
import torch.nn as nn
import torch.nn.functional as F

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
