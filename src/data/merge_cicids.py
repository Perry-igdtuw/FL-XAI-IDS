from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw/cicids2017")
OUT_PATH = Path("data/processed/cicids2017_merged.csv")

def read_and_clean_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Strip spaces in column names (CICIDS often has leading spaces)
    df.columns = [c.strip() for c in df.columns]

    # Drop duplicate columns (and ".1" variants if base exists)
    df = df.loc[:, ~df.columns.duplicated()]
    drop_cols = []
    for c in df.columns:
        if "." in c:
            base = c.split(".")[0]
            if base in df.columns:
                drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # Ensure Label exists
    if "Label" not in df.columns:
        raise ValueError(f"'Label' column not found in {path.name}. Found columns: {df.columns.tolist()[:10]}")

    # Strip label whitespace
    df["Label"] = df["Label"].astype(str).str.strip()

    return df

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {RAW_DIR}. Create it and put CICIDS CSVs inside.")

    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}.")

    frames = []
    for p in csvs:
        print(f"Reading: {p.name}")
        frames.append(read_and_clean_one(p))

    df = pd.concat(frames, ignore_index=True)

    # Replace inf and keep NaN for later fill
    df = df.replace([np.inf, -np.inf], np.nan)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nMerged rows: {len(df):,}")
    print("Label distribution (top 20):")
    print(df["Label"].value_counts().head(20))
    print(f"\nSaved merged dataset: {OUT_PATH}")

if __name__ == "__main__":
    main()