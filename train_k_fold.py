import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error
from tqdm import tqdm
import random

# ======== Dataset Definition ======== #
class HbHistogramDataset(Dataset):
    def __init__(self, metadata_csv):
        self.samples = []
        df = pd.read_csv(metadata_csv)

        df = df.dropna(subset=['hb', 'csv_path'])
        df = df[(df['csv_path'].str.strip() != '') & (df['hb'].astype(str).str.strip() != '')]

        for _, row in df.iterrows():
            csv_path = row['csv_path']
            hb_value = row['hb']

            if not os.path.exists(csv_path):
                continue

            try:
                data = pd.read_csv(csv_path, header=None).values
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

# ======== Model Definition ======== #
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
        self.block4 = ResidualBlock(256, 128, dropout_rate)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.output(x)

# ======== Loss Functions ======== #
class LogCoshLoss(nn.Module):
    def forward(self, pred, target):
        loss = torch.log(torch.cosh(pred - target + 1e-12))
        return torch.mean(loss)

# ======== Evaluation ======== #
def evaluate_model(model, test_loader, k=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).squeeze().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

    r2 = r2_score(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    maxerr = max_error(all_labels, all_preds)

    if k > 0:
        total_samples = len(all_labels)
        k = min(k, total_samples)
        indices = random.sample(range(total_samples), k)
        for i, idx in enumerate(indices, 1):
            print(f"{i}. True: {all_labels[idx]:.2f}, Predicted: {all_preds[idx]:.2f}")

    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Max Error: {maxerr:.4f}")

    return {'r2': r2, 'mse': mse, 'mae': mae, 'max_error': maxerr}

# ======== Cross Validation ======== #
def cross_validate_model(dataset_csv, k_folds=5, epochs=100, batch_size=64, lr=1e-3, weight_decay=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    full_dataset = HbHistogramDataset(dataset_csv)
    print(f"Dataset size: {len(full_dataset)} samples")

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\nFold {fold+1}/{k_folds}")
        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size)

        model = DeepHbNet().to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = LogCoshLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

        best_val_loss = float('inf')
        best_model_path = f'best_model_fold_{fold+1}.pth'

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                optimizer.zero_grad()
                preds = model(X)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step(total_loss)

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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

        model.load_state_dict(torch.load(best_model_path))
        metrics = evaluate_model(model, test_loader, k=0)
        fold_results.append(metrics)

    print("\nCross-Validation Results")
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    avg_mse = np.mean([r['mse'] for r in fold_results])
    avg_mae = np.mean([r['mae'] for r in fold_results])
    avg_maxerr = np.mean([r['max_error'] for r in fold_results])

    print(f"Average R2: {avg_r2:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average Max Error: {avg_maxerr:.4f}")

    return fold_results

if __name__ == "__main__":
    dataset_csv = "processed_features_new/updated_dataset.csv"
    cross_validate_model(dataset_csv)
