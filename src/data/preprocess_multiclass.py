from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

IN_PATH = Path("data/processed/cicids2017_merged.csv")
OUT_PATH = Path("data/processed/cicids2017_clean.csv")
LABEL_COL = "Label"

def main():
    df = pd.read_csv(IN_PATH, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

    # Separate X/y
    y = df[LABEL_COL]
    X = df.drop(columns=[LABEL_COL])

    # Convert all features to numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Handle inf/NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Reattach labels + encode
    out = X.copy()
    out[LABEL_COL] = y

    classes = sorted(out[LABEL_COL].unique().tolist())
    label2id = {c: i for i, c in enumerate(classes)}
    out["LabelId"] = out[LABEL_COL].map(label2id).astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Num classes:", len(classes))
    print("Classes:", classes)
    print("Top label counts:\n", out[LABEL_COL].value_counts().head(15))

if __name__ == "__main__":
    main()