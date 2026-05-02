from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

IN_PATH = Path("data/processed/splits_cicids2017_fl/train.csv")
OUT_ROOT = Path("data/clients/cicids2017_fl")
NUM_CLIENTS = 10
SEED = 42

def save_client_splits(client_dfs: list[pd.DataFrame], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, cdf in enumerate(client_dfs):
        client_dir = out_dir / f"client_{i:02d}"
        client_dir.mkdir(parents=True, exist_ok=True)
        cdf.to_csv(client_dir / "train.csv", index=False)

def iid_partition(df: pd.DataFrame, num_clients: int, seed: int) -> list[pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Split indices instead of dataframe directly
    indices = np.array_split(np.arange(len(df)), num_clients)

    return [df.iloc[idx].reset_index(drop=True) for idx in indices]

def label_skew_partition(df: pd.DataFrame, num_clients: int, seed: int, alpha: float = 0.5) -> list[pd.DataFrame]:
    """
    Dirichlet-based non-IID partition.
    Lower alpha => stronger label skew.
    """
    rng = np.random.default_rng(seed)
    client_parts = [[] for _ in range(num_clients)]

    for label_id, group in df.groupby("LabelId"):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        proportions = rng.dirichlet([alpha] * num_clients)
        counts = (proportions * len(group)).astype(int)

        # Fix rounding
        diff = len(group) - counts.sum()
        for j in range(diff):
            counts[j % num_clients] += 1

        start = 0
        for client_idx, count in enumerate(counts):
            if count > 0:
                client_parts[client_idx].append(group.iloc[start:start+count])
            start += count

    client_dfs = []
    for parts in client_parts:
        if parts:
            client_dfs.append(pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True))
        else:
            client_dfs.append(pd.DataFrame(columns=df.columns))
    return client_dfs

def print_partition_stats(client_dfs: list[pd.DataFrame], name: str):
    print(f"\n=== {name} partition stats ===")
    for i, cdf in enumerate(client_dfs):
        print(f"\nclient_{i:02d}: n={len(cdf)}")
        if len(cdf) > 0:
            print(cdf["Label"].value_counts().head(10))

def main():
    df = pd.read_csv(IN_PATH, low_memory=False)

    iid_clients = iid_partition(df, NUM_CLIENTS, SEED)
    iid_dir = OUT_ROOT / "iid"
    save_client_splits(iid_clients, iid_dir)
    print_partition_stats(iid_clients, "IID")

    noniid_clients = label_skew_partition(df, NUM_CLIENTS, SEED, alpha=0.3)
    noniid_dir = OUT_ROOT / "label_skew_alpha_0_3"
    save_client_splits(noniid_clients, noniid_dir)
    print_partition_stats(noniid_clients, "Non-IID label-skew")

    print("\nSaved client partitions under:", OUT_ROOT)

if __name__ == "__main__":
    main()