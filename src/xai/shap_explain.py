from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import torch
import matplotlib.pyplot as plt

from src.models.mlp import MLP


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path: str, input_dim: int, num_classes: int):
    model = MLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[256, 128, 64],
        dropout=0.2,
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model
        

def predict_fn(model, x_np: np.ndarray):
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32).to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def load_json_or_meta(path: str, key: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    return obj


def compute_global_importance(shap_values_obj) -> np.ndarray:
    """
    Return mean absolute SHAP importance per feature.
    Handles binary/multiclass shapes robustly.
    """
    vals = shap_values_obj.values

    # Common shapes:
    # binary/regression: (n_samples, n_features)
    # multiclass: (n_samples, n_features, n_classes)
    # sometimes: (n_samples, n_classes, n_features)
    if vals.ndim == 2:
        importance = np.abs(vals).mean(axis=0)

    elif vals.ndim == 3:
        # Try to infer which axis is features
        # We know X_explain shape later is (n_samples, n_features)
        # Most often SHAP returns (n_samples, n_features, n_classes)
        # so averaging over samples and classes leaves features
        importance = np.abs(vals).mean(axis=(0, 2))

        # Safety fallback if feature length looks wrong will be checked later
    else:
        raise ValueError(f"Unsupported SHAP values shape: {vals.shape}")

    return importance


def explain_model(
    model_path: str,
    split_csv: str,
    scaler_path: str,
    feature_cols_path: str,
    label_names_path: str,
    out_dir: str,
    background_size: int = 256,
    explain_size: int = 100,
    top_k: int = 20,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(split_csv, low_memory=False)
    scaler = joblib.load(scaler_path)

    feature_cols = load_json_or_meta(feature_cols_path, "feature_cols")
    label_names = load_json_or_meta(label_names_path, "label_names")

    # Sample manageable subset
    n_take = min(len(df), background_size + explain_size)
    df = df.sample(n=n_take, random_state=42).reset_index(drop=True)

    X = df[feature_cols].copy()
    X[feature_cols] = scaler.transform(X[feature_cols])

    X_bg = X.iloc[: min(background_size, len(X))].values.astype("float32")
    start_idx = min(background_size, len(X))
    end_idx = min(background_size + explain_size, len(X))
    X_explain = X.iloc[start_idx:end_idx].values.astype("float32")

    if len(X_explain) == 0:
        raise ValueError("X_explain is empty. Increase dataset size or reduce background_size.")

    model = load_model(model_path, input_dim=len(feature_cols), num_classes=len(label_names))

    explainer = shap.Explainer(lambda z: predict_fn(model, z), X_bg)
    shap_values = explainer(X_explain)

    # 1) Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_explain, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Robust manual global bar plot
    importance = compute_global_importance(shap_values)

    if len(importance) != len(feature_cols):
        raise ValueError(
            f"Importance length mismatch: got {len(importance)} values for {len(feature_cols)} features."
        )

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    imp_df.to_csv(out / "global_feature_importance.csv", index=False)

    top_df = imp_df.head(top_k).iloc[::-1]  # reverse for horizontal bar chart

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig(out / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP outputs to: {out}")


if __name__ == "__main__":
    # Centralized
    explain_model(
        model_path="results/centralized/best_model.pt",
        split_csv="data/processed/splits_cicids2017_fl/test.csv",
        scaler_path="results/centralized/scaler.joblib",
        feature_cols_path="results/centralized/feature_cols.json",
        label_names_path="results/centralized/label_names.json",
        out_dir="results/centralized/shap",
    )

    # FL-IID
    explain_model(
        model_path="results/fl/iid/global_model.pt",
        split_csv="data/processed/splits_cicids2017_fl/test.csv",
        scaler_path="results/fl/global_scaler.joblib",
        feature_cols_path="results/fl/metadata.json",
        label_names_path="results/fl/metadata.json",
        out_dir="results/fl/iid/shap",
    )

    # FL-Non-IID
    explain_model(
        model_path="results/fl/label_skew_alpha_0_3/global_model.pt",
        split_csv="data/processed/splits_cicids2017_fl/test.csv",
        scaler_path="results/fl/global_scaler.joblib",
        feature_cols_path="results/fl/metadata.json",
        label_names_path="results/fl/metadata.json",
        out_dir="results/fl/label_skew_alpha_0_3/shap",
    )