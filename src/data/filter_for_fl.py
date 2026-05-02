from __future__ import annotations
from pathlib import Path
import pandas as pd

IN_PATH = Path("data/processed/cicids2017_clean.csv")
OUT_PATH = Path("data/processed/cicids2017_fl_ready.csv")
MIN_SAMPLES = 1000

def main():
    df = pd.read_csv(IN_PATH, low_memory=False)

    counts = df["Label"].value_counts()
    keep_labels = counts[counts >= MIN_SAMPLES].index.tolist()

    df = df[df["Label"].isin(keep_labels)].copy()

    # Re-encode LabelId after filtering
    classes = sorted(df["Label"].unique().tolist())
    label2id = {c: i for i, c in enumerate(classes)}
    df["LabelId"] = df["Label"].map(label2id).astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("Saved:", OUT_PATH)
    print("Num classes:", len(classes))
    print("Classes:", classes)
    print("\nLabel distribution:")
    print(df["Label"].value_counts())

if __name__ == "__main__":
    main()