import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error
import random
# ======== Dataset Definition ======== #
class HbHistogramDataset(Dataset):
    def __init__(self, metadata_csv):
        self.samples = []
        df = pd.read_csv(metadata_csv)

        # Drop rows with missing or empty values
        df = df.dropna(subset=['hb', 'csv_path'])
        df = df[(df['csv_path'].str.strip() != '') & (df['hb'].astype(str).str.strip() != '')]

        for _, row in df.iterrows():
            csv_path = row['csv_path']
            hb_value = row['hb']

            if not os.path.exists(csv_path):
                continue

            try:
                data = pd.read_csv(csv_path, header=None).values  # (20, 2304)
                for i in range(data.shape[0]):
                    self.samples.append((data[i], hb_value))
            except Exception as e:
                print(f"[Error] reading {csv_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label = self.samples[idx]
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return features, label

# ======== Deep Residual Model Definition with Dropout ======== #
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

class DeepHbNet(nn.Module):
    def __init__(self, input_size=2304, dropout_rate=0.2):
        super(DeepHbNet, self).__init__()
        self.block1 = ResidualBlock(input_size, 1024, dropout_rate)
        self.block2 = ResidualBlock(1024, 512, dropout_rate)
        self.block3 = ResidualBlock(512, 256, dropout_rate)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.output(x)

# ======== Custom Loss Functions (Optional) ======== #
class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        loss = torch.log(torch.cosh(pred - target + 1e-12))
        return torch.mean(loss)

class TukeyLoss(nn.Module):
    def __init__(self, c=4.685):
        super().__init__()
        self.c = c

    def forward(self, pred, target):
        error = pred - target
        mask = torch.abs(error) < self.c
        loss = torch.zeros_like(error)
        loss[mask] = ((self.c**2) / 6) * (1 - (1 - (error[mask] / self.c) ** 2) ** 3)
        loss[~mask] = (self.c**2) / 6
        return loss.mean()

# ======== Training Function ======== #
def train_model(dataset_csv, epochs=80, batch_size=64, lr=1e-3, weight_decay=1e-4, save_dir='./best_model'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    full_dataset = HbHistogramDataset(dataset_csv)
    total = len(full_dataset)
    train_len = int(0.8 * total)
    test_len = total - train_len
    print(f"📦 Dataset size: {total} samples → Train: {train_len}, Test: {test_len}")

    train_set, test_set = random_split(full_dataset, [train_len, test_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = DeepHbNet().to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.MSELoss()  # or LogCoshLoss() or TukeyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_val_loss = float('inf')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss)

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                preds = model(X)
                val_loss += criterion(preds, y).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("✅ Validation loss improved, saving model...")
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    # Load best model before returning
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    return model, test_loader

# ======== Evaluation Function ======== #
def evaluate_model(model, test_loader, k=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    # Evaluation metrics
    r2 = r2_score(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    maxerr = max_error(all_labels, all_preds)

    # Select k random predictions
    total_samples = len(all_labels)
    k = min(k, total_samples)
    indices = random.sample(range(total_samples), k)
    random_predictions = [(all_labels[i], all_preds[i]) for i in indices]

    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Max Error: {maxerr:.4f}")
    print(f"\n{k} Random Predictions (True, Predicted):")
    for i, (true_val, pred_val) in enumerate(random_predictions, 1):
        print(f"{i}. True: {true_val:.2f}, Predicted: {pred_val:.2f}")

    return {
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'max_error': maxerr,
        'random_predictions': random_predictions
    }

# ======== Entry Point ======== #
if __name__ == "__main__":
    dataset_csv = "processed_features_new/updated_dataset.csv"  # <-- Change this if needed
    model, test_loader = train_model(dataset_csv)
    evaluate_model(model, test_loader)
