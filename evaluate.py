import torch
from torch.utils.data import DataLoader
from train import DeepHbNet, evaluate_model, HbHistogramDataset

# ===== Configuration ===== #
CSV_PATH = "processed_features/updated_dataset.csv"
MODEL_PATH = "best_model/best_model_99.pth"
BATCH_SIZE = 2

# ===== Prepare Dataset ===== #
dataset = HbHistogramDataset(CSV_PATH)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=2)

# ===== Load Model ===== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepHbNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print("✅ Model loaded successfully.")

# ===== Evaluate Model ===== #
evaluate_model(model, test_loader)
