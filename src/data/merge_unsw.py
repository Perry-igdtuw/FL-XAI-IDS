from __future__ import annotations
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/unsw")
TRAIN_PATH = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_PATH = RAW_DIR / "UNSW_NB15_testing-set.csv"
OUT_PATH = Path("data/processed/unsw_merged.csv")


def read_unsw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw UNSW directory not found: {RAW_DIR}")

    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Expected UNSW files not found in {RAW_DIR}.\n"
            f"Found: {[p.name for p in sorted(RAW_DIR.glob('*.csv'))]}"
        )

    train_df = read_unsw_csv(TRAIN_PATH)
    test_df = read_unsw_csv(TEST_PATH)

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("Training and testing CSV columns do not match for UNSW.")

    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUT_PATH, index=False)

    print(f"Saved merged UNSW dataset to: {OUT_PATH}")
    print(f"Rows: {len(merged_df):,}")
    print("Columns:", len(merged_df.columns))
    print("Label columns:", [c for c in merged_df.columns if c.lower() in {"attack_cat", "label"}])


if __name__ == "__main__":
    main()
