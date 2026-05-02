from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from src.models.mlp import MLP

GLOBAL_SPLIT_DIR = Path("data/processed/splits_cicids2017_fl")
CLIENT_ROOT = Path("data/clients/cicids2017_fl")
FL_RESULTS_DIR = Path("results/fl")
FL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = FL_RESULTS_DIR / "metadata.json"
SCALER_PATH = FL_RESULTS_DIR / "global_scaler.joblib"


def prepare_global_metadata() -> dict:
    train_df = pd.read_csv(GLOBAL_SPLIT_DIR / "train.csv", low_memory=False)

    feature_cols = [c for c in train_df.columns if c not in ["Label", "LabelId"]]
    label_names = (
        train_df[["LabelId", "Label"]]
        .drop_duplicates()
        .sort_values("LabelId")["Label"]
        .tolist()
    )

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    joblib.dump(scaler, SCALER_PATH)

    meta = {
        "feature_cols": feature_cols,
        "label_names": label_names,
        "num_classes": len(label_names),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def load_metadata() -> dict:
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_global_scaler():
    return joblib.load(SCALER_PATH)


def load_client_df(partition_name: str, client_id: int) -> pd.DataFrame:
    path = CLIENT_ROOT / partition_name / f"client_{client_id:02d}" / "train.csv"
    return pd.read_csv(path, low_memory=False)


def make_model(input_dim: int, num_classes: int) -> MLP:
    return MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    )


def get_model_parameters(model):
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def set_model_parameters(model, parameters):
    import torch
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(new_state, strict=True)