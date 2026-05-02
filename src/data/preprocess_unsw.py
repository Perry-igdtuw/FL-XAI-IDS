from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.data.preprocess_multiclass import clean_and_encode

IN_PATH = Path("data/processed/unsw_merged.csv")
OUT_PATH = Path("data/processed/unsw_clean.csv")
LABEL_COL = "attack_cat"


def main():
    df = pd.read_csv(IN_PATH, low_memory=False)
    clean_and_encode(df, LABEL_COL, OUT_PATH)


if __name__ == "__main__":
    main()
