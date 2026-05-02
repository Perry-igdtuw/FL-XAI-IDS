from __future__ import annotations
import json
from pathlib import Path

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.mlp import MLP
from src.data.dataset import TabularDataset
from src.models.train_utils import predict


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str, input_dim: int, num_classes: int) -> MLP:
    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    )
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def save_predictions(
    model_path: str,
    split_csv: str,
    scaler_path: str,
    feature_cols_path: str,
    label_names_path: str,
    out_csv: str,
) -> None:
    df = pd.read_csv(split_csv, low_memory=False)

    scaler = joblib.load(scaler_path)

    def _load_json_or_meta(path: str, key: str):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and key in obj:
            return obj[key]
        return obj

    # later inside save_predictions(...)
    feature_cols = _load_json_or_meta(feature_cols_path, "feature_cols")
    label_names = _load_json_or_meta(label_names_path, "label_names")

    num_classes = len(label_names)

    # Scale
    df[feature_cols] = scaler.transform(df[feature_cols])

    ds = TabularDataset(df, feature_cols, "LabelId")
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)

    model = load_model(model_path, input_dim=len(feature_cols), num_classes=num_classes)
    y_true, y_pred = predict(model, loader, DEVICE)

    out_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_true_label": [label_names[i] for i in y_true],
            "y_pred_label": [label_names[i] for i in y_pred],
        }
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    # Centralized
    save_predictions(
        model_path="results/centralized/best_model.pt",
        split_csv="data/processed/splits_cicids2017_fl/test.csv",
        scaler_path="results/centralized/scaler.joblib",
        feature_cols_path="results/centralized/feature_cols.json",
        label_names_path="results/centralized/label_names.json",
        out_csv="results/centralized/test_predictions.csv",
    )

    # FL-IID
    save_predictions(
        model_path="results/fl/iid/global_model.pt",
        split_csv="data/processed/splits_cicids2017_fl/test.csv",
        scaler_path="results/fl/global_scaler.joblib",
        feature_cols_path="results/fl/metadata.json",
        label_names_path="results/fl/metadata.json",
        out_csv="results/fl/iid/test_predictions.csv",
    )