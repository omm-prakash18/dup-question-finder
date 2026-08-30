"""
src/data/inspect_data.py
─────────────────────────────────────────────────────────────────────────────
DATA SANITY INSPECTION SCRIPT
Samples N random pairs from the dataset, prints them cleanly formatted,
and summarizes key data quality statistics.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

PROCESSED_DIR = Path(CFG["paths"]["processed_data"])


def inspect_dataset(split: str = "train", sample_n: int = 10, seed: int = 42):
    file_path = PROCESSED_DIR / f"{split}.parquet"
    if not file_path.exists():
        print(f"File {file_path} not found. Run download_data.py first.")
        return

    df = pd.read_parquet(file_path)
    print(f"\n" + "="*70)
    print(f" DATASET INSPECTION: Split='{split}' | Total Rows: {len(df):,}")
    print(f"="*70)

    # Label statistics
    dup_count = df["is_duplicate"].sum()
    non_dup_count = len(df) - dup_count
    print(f"Class Balance: Duplicate = {dup_count:,} ({dup_count/len(df):.1%}) | Non-Duplicate = {non_dup_count:,} ({non_dup_count/len(df):.1%})")

    # Sample random pairs
    sampled = df.sample(n=min(sample_n, len(df)), random_state=seed)

    print(f"\n--- Random Sample of {len(sampled)} Pairs ---")
    for idx, row in sampled.iterrows():
        label_str = "DUPLICATE (1)" if row["is_duplicate"] == 1 else "NON-DUPLICATE (0)"
        print(f"\n[Row {idx}] Label: {label_str}")
        print(f"  Q1: {row['question1']}")
        print(f"  Q2: {row['question2']}")
    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect random question pairs.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=10, help="Number of pairs to print")
    args = parser.parse_args()

    inspect_dataset(split=args.split, sample_n=args.n)
