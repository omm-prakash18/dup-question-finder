"""
tests/test_hybrid_search.py — Hybrid Search (BM25 + SBERT + RRF) Tests
─────────────────────────────────────────────────────────────────────────────
PRODUCTION FAILURE MODE:
  - Vector search misses exact keyword errors (e.g. "0x80070005" or "pd.read_parquet").
  - Mismatched tokenizers between index time and query time cause zero matches.
  - Off-by-one errors in Reciprocal Rank Fusion (RRF) math distort rank order.
"""

import pytest
from src.models.hybrid_search import SimpleBM25, reciprocal_rank_fusion, default_bm25_tokenizer


@pytest.fixture
def sample_corpus():
    return [
        {"question_id": 1, "question_text": "How do I fix error 0x80070005 in Windows Update?"},
        {"question_id": 2, "question_text": "How to read CSV files using pandas pd.read_csv?"},
        {"question_id": 3, "question_text": "How to load parquet file with pd.read_parquet in Python?"},
        {"question_id": 4, "question_text": "General machine learning tutorial for beginners."},
    ]


def test_bm25_exact_keyword_retrieval(sample_corpus):
    """
    Verify BM25 successfully retrieves exact keyword/error code matches (0x80070005).
    """
    bm25 = SimpleBM25(sample_corpus)
    results = bm25.search("Windows error 0x80070005", top_k=2)

    assert len(results) > 0
    assert results[0]["question_id"] == 1
    assert "0x80070005" in results[0]["question_text"]


def test_reciprocal_rank_fusion_math():
    """
    Verify RRF formula output matches manual calculation:
    RRF_score = 1 / (60 + r_dense) + 1 / (60 + r_bm25)
    """
    dense_list = [
        {"question_id": 10, "question_text": "Doc A"},  # Rank 1
        {"question_id": 20, "question_text": "Doc B"},  # Rank 2
    ]
    bm25_list = [
        {"question_id": 20, "question_text": "Doc B"},  # Rank 1
        {"question_id": 10, "question_text": "Doc A"},  # Rank 2
    ]

    fused = reciprocal_rank_fusion([dense_list, bm25_list], k=60, top_k=2)

    # Both Doc A and Doc B have equal reciprocal rank sums: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    expected_score = round(1.0 / 61 + 1.0 / 62, 6)

    assert len(fused) == 2
    assert fused[0]["score"] == expected_score
    assert fused[1]["score"] == expected_score


def test_rrf_degrades_gracefully_on_empty_retriever():
    """
    When one retriever returns zero results, RRF should degrade gracefully without error.
    """
    dense_list = [{"question_id": 100, "question_text": "Doc X"}]
    bm25_list = []

    fused = reciprocal_rank_fusion([dense_list, bm25_list], k=60, top_k=5)

    assert len(fused) == 1
    assert fused[0]["question_id"] == 100


def test_tokenizer_consistency():
    """
    Verify tokenizer used for indexing and query produces identical tokens for error code.
    """
    text = "Error 0x80070005 in pd.read_parquet"
    tokens1 = default_bm25_tokenizer(text)
    tokens2 = default_bm25_tokenizer(text)

    assert tokens1 == tokens2
    assert "0x80070005" in tokens1
    assert "read_parquet" in tokens1
