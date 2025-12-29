
# In[1]:
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from utils import ScorerStepByStep, DataPoint

#  Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#  Dataset
class SequenceDataset(Dataset):
    def __init__(self, data, window_size=10, warmup=100, n_jobs=-1):
        self.window_size = window_size
        self.warmup = warmup
        self.n_jobs = n_jobs
        self.samples = self._create_samples(data)

    def _process_sequence(self, seq_data):
        seq_id, seq_df = seq_data
        seq_values = seq_df.iloc[:, 3:].values
        samples = []
        for t in range(self.warmup, len(seq_values) - 1):
            start = t - self.window_size
            if start < 0:
                continue
            X = seq_values[start:t]
            y = seq_values[t + 1]
            samples.append((X, y))
        return samples

    def _create_samples(self, data):
        groups = list(data.groupby("seq_ix"))
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_sequence)(g) for g in groups
        )
        samples = [s for seq in results for s in seq]
        print(f"Created {len(samples)} samples.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

#  GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=None, dropout=0.15):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim or input_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        x = self.layer_norm(last)
        x = self.dropout(x)
        return self.fc(x)


#  Data Helpers
def split_data_by_sequence(df, seed):
    ids = df["seq_ix"].unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    n_train = int(len(ids) * 0.8)
    train_df = df[df["seq_ix"].isin(shuffled[:n_train])]
    val_df = df[df["seq_ix"].isin(shuffled[n_train:])]
    print(f"Train sequences: {len(shuffled[:n_train])}, Validation sequences: {len(shuffled[n_train:])}")
    return train_df, val_df


def create_loaders(train_set, val_set, batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

#  Training and Evaluation
def train_one_epoch(model, loader, criterion, optimizer, clip, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.size(0)
            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    val_loss = total_loss / len(loader.dataset)
    r2 = r2_score(y_true, y_pred)
    return val_loss, r2


#  Training Entry
def train_model(config):
    set_seed(config["seed"])

    df = pd.read_parquet(config["paths"]["dataset"])
    window_size = config["training"]["window_size"]
    warmup = config["training"]["warmup"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    wd = config["training"]["weight_decay"]
    clip = config["regularization"]["gradient_clip"]
    patience = config["regularization"]["scheduler_patience"]
    factor = config["regularization"]["scheduler_factor"]
    device = config["device"]

    n_features = len(df.columns) - 3
    train_df, val_df = split_data_by_sequence(df, config["seed"])
    train_set = SequenceDataset(train_df, window_size, warmup)
    val_set = SequenceDataset(val_df, window_size, warmup)
    train_loader, val_loader = create_loaders(train_set, val_set, batch_size, config["seed"])

    model = GRUModel(
        input_dim=n_features,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"]
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=factor, patience=patience)

    best_val = float("inf")
    best_r2 = -1.0
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, clip, device)
        val_loss, val_r2 = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val R²: {val_r2:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_r2 = val_r2
            best_state = model.state_dict().copy()
            print("New best model found.")

    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "window_size": window_size,
        "dropout": config["model"]["dropout"]
    }, config["paths"]["model_checkpoint"])
    print(f"Model saved to {config['paths']['model_checkpoint']}")
    print(f"Best Val Loss: {best_val:.6f}, Best Val R²: {best_r2:.6f}")

    from solution import PredictionModel
    scorer = ScorerStepByStep(val_df)
    prediction_model = PredictionModel(model, window_size, device)
    results = scorer.score(prediction_model)
    print(f"Final Mean R² (Scorer): {results['mean_r2']:.6f}")

#  Main
if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    train_model(config)


