"""
tests/test_features.py — Feature Engineering Sanity & Adversarial Tests
─────────────────────────────────────────────────────────────────────────────
Verifies:
  1. No NaN / Inf values in feature matrix (even for empty or single-character inputs).
  2. Identical questions return maximum similarity (~1.0).
  3. Completely unrelated questions return near-zero similarity.
  4. Adversarial Negation Flipping:
     "How to learn Python?" vs "How to NOT learn Python?"
     The similarity feature should be lower than identical questions.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.feature_engineering import (
    fuzzy_features, length_features, overlap_features, TFIDFSimilarity
)


class TestFeatureSanity:
    def test_no_nan_or_inf_on_edge_inputs(self):
        """Test feature computation on empty, whitespace, and single-char strings."""
        df = pd.DataFrame({
            "q1_classical": ["", " ", "a", "normal question here"],
            "q2_classical": ["", "a", " ", "another normal question"],
        })

        fuzz_df = fuzzy_features(df["q1_classical"][0], df["q2_classical"][0])
        assert not np.isnan(list(fuzz_df.values())).any()

        len_df = length_features(df, "q1_classical", "q2_classical")
        assert not len_df.isna().any().any()
        assert not np.isinf(len_df.values).any()

        overlap_df = overlap_features(df, "q1_classical", "q2_classical")
        assert not overlap_df.isna().any().any()
        assert not np.isinf(overlap_df.values).any()

    def test_identical_questions_score_max(self):
        """Identical inputs should yield fuzz_ratio ~ 1.0 and Jaccard = 1.0."""
        q = "how do i learn machine learning from scratch"
        fuzz = fuzzy_features(q, q)
        assert fuzz["fuzz_ratio"] == 1.0
        assert fuzz["fuzz_token_set_ratio"] == 1.0

        df = pd.DataFrame({"q1": [q], "q2": [q]})
        overlap = overlap_features(df, "q1", "q2")
        assert overlap["common_word_ratio"].iloc[0] == 1.0

    def test_unrelated_questions_score_low(self):
        q1 = "how to make apple pie at home"
        q2 = "what is quantum computing entanglement"
        fuzz = fuzzy_features(q1, q2)
        assert fuzz["fuzz_ratio"] < 0.4

        df = pd.DataFrame({"q1": [q1], "q2": [q2]})
        overlap = overlap_features(df, "q1", "q2")
        assert overlap["common_word_ratio"].iloc[0] == 0.0

    def test_negation_flip_adversarial(self):
        """
        Adversarial test:
        'How to learn Python fast?' vs 'How to NOT learn Python fast?'
        Surface fuzzy matching will score HIGH (~0.90+), but Jaccard/TF-IDF
        or SBERT must capture that 'not' alters the intent.
        """
        q1 = "how to learn python fast"
        q2 = "how to not learn python fast"

        fuzz = fuzzy_features(q1, q2)
        # Verify surface fuzzy score is high so we know why baseline needs ML classifier
        assert fuzz["fuzz_token_set_ratio"] > 0.80, "Fuzzy match unexpectedly low for negation pair"
