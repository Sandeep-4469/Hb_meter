import torch
import torch.nn as nn
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
        out = self.dropout(out)  # Apply dropout
        out = self.bn2(self.linear2(out))
        out += identity
        return F.relu(out)


class DeepHbNet(nn.Module):
    def __init__(self, input_size=2304, dropout_rate=0.2):
        super(DeepHbNet, self).__init__()
        self.block1 = ResidualBlock(input_size, 1024, dropout_rate)
        self.block2 = ResidualBlock(1024, 512, dropout_rate)
        self.block3 = ResidualBlock(512, 256, dropout_rate)
        self.block4 = ResidualBlock(256, 128, dropout_rate)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.output(x)

MODEL_PATH = "best_model_99.pth"
BATCH_SIZE = 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepHbNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print("✅ Model loaded successfully.")

