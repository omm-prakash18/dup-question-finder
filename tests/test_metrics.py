"""
tests/test_metrics.py — Unit tests for retrieval evaluation metrics
"""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import recall_at_k, mean_reciprocal_rank


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = [[1, 2, 3]]
        relevant  = [[2]]
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_zero_recall(self):
        retrieved = [[1, 2, 3]]
        relevant  = [[99]]
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_recall(self):
        retrieved = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        relevant  = [[2], [99], [9]]
        # 2 out of 3 queries have a hit
        assert abs(recall_at_k(retrieved, relevant, k=3) - 2/3) < 1e-6

    def test_k_cutoff(self):
        retrieved = [[1, 2, 3]]
        relevant  = [[3]]
        assert recall_at_k(retrieved, relevant, k=2) == 0.0  # relevant is at pos 3
        assert recall_at_k(retrieved, relevant, k=3) == 1.0


class TestMRR:
    def test_first_rank(self):
        retrieved = [[1, 2, 3]]
        relevant  = [[1]]
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_rank(self):
        retrieved = [[1, 2, 3]]
        relevant  = [[2]]
        assert abs(mean_reciprocal_rank(retrieved, relevant) - 0.5) < 1e-6

    def test_no_match(self):
        retrieved = [[1, 2, 3]]
        relevant  = [[99]]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0

    def test_average_multiple_queries(self):
        retrieved = [[1, 2, 3], [4, 5, 6]]
        relevant  = [[1], [5]]  # ranks: 1 and 2
        # MRR = (1/1 + 1/2) / 2 = 0.75
        assert abs(mean_reciprocal_rank(retrieved, relevant) - 0.75) < 1e-6
