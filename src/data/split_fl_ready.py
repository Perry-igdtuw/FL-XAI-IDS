from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

IN_PATH = Path("data/processed/cicids2017_fl_ready.csv")
OUT_DIR = Path("data/processed/splits_cicids2017_fl")
LABEL_ID = "LabelId"

def main(seed=42, test_size=0.15, val_size=0.15):
    df = pd.read_csv(IN_PATH, low_memory=False)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[LABEL_ID],
    )

    val_frac = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_frac,
        random_state=seed,
        stratify=train_df[LABEL_ID],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)

    print("Saved filtered FL splits to:", OUT_DIR)

if __name__ == "__main__":
    main()