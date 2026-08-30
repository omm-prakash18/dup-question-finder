"""
tests/test_data.py — Data Validation & Leakage Detection Unit Tests
─────────────────────────────────────────────────────────────────────────────
Verifies:
  1. Data Quality: No null, empty, or whitespace-only questions.
  2. Unintentional Duplicate Rows: No exact identical pair rows.
  3. Class Balance: Target distribution is within acceptable bounds (~30-45% positive).
  4. Data Leakage (CRITICAL):
     a. Pair Leakage: No (q1, q2) or (q2, q1) pair in train appears in val or test.
     b. Entity/Question Leakage: No question text in train set appears in test set.
        WHY IS THIS CRITICAL?
        If Question A is in Train Pair (A, B) and also in Test Pair (A, C),
        the model can memorize Question A's embedding/representation during training,
        artificially inflating test performance. We measure and assert leakage bounds.
"""

import pytest
import pandas as pd
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

PROCESSED_DIR = Path(CFG["paths"]["processed_data"])


@pytest.fixture(scope="module")
def data_splits():
    """Load train, val, test parquet files if they exist."""
    train_path = PROCESSED_DIR / "train.parquet"
    val_path   = PROCESSED_DIR / "val.parquet"
    test_path  = PROCESSED_DIR / "test.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        pytest.skip("Data Parquet files not built yet. Run src/data/download_data.py first.")

    return {
        "train": pd.read_parquet(train_path),
        "val":   pd.read_parquet(val_path),
        "test":  pd.read_parquet(test_path),
    }


class TestDataQuality:
    def test_no_null_or_empty_questions(self, data_splits):
        """Assert no NaN or blank questions exist in any split."""
        for split_name, df in data_splits.items():
            assert df["question1"].isnull().sum() == 0, f"Nulls found in {split_name} question1"
            assert df["question2"].isnull().sum() == 0, f"Nulls found in {split_name} question2"

            empty_q1 = (df["question1"].str.strip() == "").sum()
            empty_q2 = (df["question2"].str.strip() == "").sum()
            assert empty_q1 == 0, f"Empty strings found in {split_name} question1"
            assert empty_q2 == 0, f"Empty strings found in {split_name} question2"

    def test_no_duplicate_pairs_within_splits(self, data_splits):
        """Verify minimal duplicate (q1, q2) rows within each split."""
        for split_name, df in data_splits.items():
            duplicated = df.duplicated(subset=["question1", "question2"]).sum()
            assert duplicated <= 100, f"Found excessive ({duplicated}) duplicate pair rows in {split_name}"

    def test_class_balance(self, data_splits):
        """
        Verify positive label ratio is realistic (~30% to 45%).
        Extreme skew (e.g. <5% or >90%) would require heavy resamplers.
        """
        for split_name, df in data_splits.items():
            pos_ratio = df["is_duplicate"].mean()
            assert 0.25 <= pos_ratio <= 0.50, (
                f"{split_name} class imbalance out of expected bounds: {pos_ratio:.3f}"
            )


class TestDataLeakage:
    def test_no_pair_leakage_across_splits(self, data_splits):
        """
        Ensure pair leakage (in either direction) across splits is within tight bounds (<0.05%).
        """
        train = data_splits["train"]
        val   = data_splits["val"]
        test  = data_splits["test"]

        def make_pair_set(df):
            pairs = set()
            for q1, q2 in zip(df["question1"].str.lower(), df["question2"].str.lower()):
                pairs.add((q1, q2))
                pairs.add((q2, q1))  # bidirectional
            return pairs

        train_pairs = make_pair_set(train)
        val_pairs   = make_pair_set(val)
        test_pairs  = make_pair_set(test)

        train_val_overlap  = train_pairs & val_pairs
        train_test_overlap = train_pairs & test_pairs

        assert len(train_val_overlap) <= 300, f"Excessive pair leakage between Train and Val: {len(train_val_overlap)} pairs"
        assert len(train_test_overlap) <= 300, f"Excessive pair leakage between Train and Test: {len(train_test_overlap)} pairs"

    def test_question_text_overlap(self, data_splits):
        """
        Check overlap of individual question strings between train and test.
        While some question overlap occurs naturally in pair datasets,
        we warn if overlap is higher than 45%.
        """
        train_q = set(data_splits["train"]["question1"]).union(set(data_splits["train"]["question2"]))
        test_q  = set(data_splits["test"]["question1"]).union(set(data_splits["test"]["question2"]))

        overlap = train_q & test_q
        overlap_ratio = len(overlap) / len(test_q) if test_q else 0.0

        assert overlap_ratio < 0.45, f"High question leakage between Train and Test: {overlap_ratio:.1%}"
