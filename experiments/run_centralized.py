from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from src.models.mlp import MLP
from src.data.dataset import TabularDataset
from src.models.train_utils import train_one_epoch, predict, make_class_weights
from src.eval.metrics import compute_metrics, per_class_report
from src.utils.seed import set_seed


DATA_DIR = Path("data/processed/splits_cicids2017_fl")
OUT_DIR = Path("results/centralized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 2048
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    set_seed(SEED)

    train_df = pd.read_csv(DATA_DIR / "train.csv", low_memory=False)
    val_df = pd.read_csv(DATA_DIR / "val.csv", low_memory=False)
    test_df = pd.read_csv(DATA_DIR / "test.csv", low_memory=False)

    feature_cols = [c for c in train_df.columns if c not in ["Label", "LabelId"]]

    # Standardize using train only
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    joblib.dump(scaler, OUT_DIR / "scaler.joblib")

    num_classes = int(train_df["LabelId"].nunique())
    label_names = (
        train_df[["LabelId", "Label"]]
        .drop_duplicates()
        .sort_values("LabelId")["Label"]
        .tolist()
    )

    train_ds = TabularDataset(train_df, feature_cols, "LabelId")
    val_ds = TabularDataset(val_df, feature_cols, "LabelId")
    test_ds = TabularDataset(test_df, feature_cols, "LabelId")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MLP(
        input_dim=len(feature_cols),
        num_classes=num_classes,
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    ).to(DEVICE)

    class_weights = make_class_weights(train_df["LabelId"].values, num_classes).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        y_val, y_val_pred = predict(model, val_loader, DEVICE)
        val_metrics = compute_metrics(y_val, y_val_pred)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        history.append(row)

        print(
            f"[Epoch {epoch}/{EPOCHS}] "
            f"loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT_DIR / "best_model.pt")

    y_test, y_test_pred = predict(model, test_loader, DEVICE)
    test_metrics = compute_metrics(y_test, y_test_pred)

    print("\nFinal Test Metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    report = per_class_report(y_test, y_test_pred, label_names)
    print("\nClassification Report:\n")
    print(report)

    pd.DataFrame(history).to_csv(OUT_DIR / "history.csv", index=False)
    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(OUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    with open(OUT_DIR / "feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    with open(OUT_DIR / "label_names.json", "w", encoding="utf-8") as f:
        json.dump(label_names, f, indent=2)


if __name__ == "__main__":
    main()