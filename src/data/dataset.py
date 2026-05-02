from __future__ import annotations
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list[str], label_col: str = "LabelId") -> None:
        self.X = df[feature_cols].values.astype("float32")
        self.y = df[label_col].values.astype("int64")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])