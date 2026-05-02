from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def make_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device: str):
    model.train()
    total_loss = 0.0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total += bs

    return total_loss / max(total, 1)


@torch.no_grad()
def predict(model, loader: DataLoader, device: str):
    model.eval()
    all_true = []
    all_pred = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()

        all_pred.extend(pred.tolist())
        all_true.extend(yb.numpy().tolist())

    return all_true, all_pred