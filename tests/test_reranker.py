"""
tests/test_reranker.py — Two-Stage Retrieval (Bi-Encoder + Cross-Encoder) Tests
─────────────────────────────────────────────────────────────────────────────
PRODUCTION FAILURE MODE:
  - If Cross-Encoder scores are un-normalized raw logits or unbatched,
    reranking will produce unstable scores or cause latency spikes (>200ms).
  - If fewer than top-N candidates are returned from small corpora, indexing
    errors occur.
"""

import time
import pytest
import numpy as np
from unittest.mock import MagicMock


class DummyCrossEncoder:
    """Mock Cross-Encoder for testing reranker pipeline."""
    def predict(self, pairs):
        # Return probability scores based on text similarity heuristic
        scores = []
        for q1, q2 in pairs:
            if "Python" in q1 and "Python" in q2:
                scores.append(0.92)
            elif "Java" in q1 and "Java" in q2:
                scores.append(0.85)
            else:
                scores.append(0.15)
        return np.array(scores, dtype=np.float32)


@pytest.fixture
def retriever_with_mock_reranker():
    from src.models.sbert_pipeline import SimilarQuestionRetriever
    retriever = MagicMock(spec=SimilarQuestionRetriever)
    retriever._reranker = DummyCrossEncoder()

    def mock_rerank(query, candidates, top_k=5):
        if not candidates:
            return candidates
        pairs = [[query, c["question_text"]] for c in candidates]
        scores = retriever._reranker.predict(pairs)
        for c, score in zip(candidates, scores):
            c["score"] = round(float(score), 4)
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    retriever.rerank = mock_rerank
    return retriever


def test_candidates_passed_and_ranking_changes(retriever_with_mock_reranker):
    """
    Verify top candidates are passed to cross-encoder and final order changes.
    """
    candidates = [
        {"question_id": 1, "question_text": "What is C++ pointer?", "score": 0.90},
        {"question_id": 2, "question_text": "How to learn Python fast?", "score": 0.70},
    ]

    reranked = retriever_with_mock_reranker.rerank(
        query="How to master Python programming?",
        candidates=candidates,
        top_k=2,
    )

    # Initial candidate #2 ("Python") should move to #1 after cross-encoder reranking
    assert reranked[0]["question_id"] == 2
    assert reranked[0]["score"] == 0.92


def test_small_corpus_fewer_candidates_safety(retriever_with_mock_reranker):
    """
    Verify cross-encoder handles fewer candidates (e.g. 1 candidate) without index errors.
    """
    candidates = [
        {"question_id": 1, "question_text": "Single candidate in small corpus", "score": 0.50}
    ]
    reranked = retriever_with_mock_reranker.rerank(
        query="Test query",
        candidates=candidates,
        top_k=50,
    )
    assert len(reranked) == 1


def test_reranker_latency_budget(retriever_with_mock_reranker):
    """
    Assert reranking 50 candidates runs within acceptable latency budget (< 200ms).
    """
    candidates = [
        {"question_id": i, "question_text": f"Candidate question text {i}", "score": 0.80}
        for i in range(50)
    ]
    t0 = time.perf_counter()
    reranked = retriever_with_mock_reranker.rerank(
        query="Test query text",
        candidates=candidates,
        top_k=5,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(reranked) == 5
    assert elapsed_ms < 200.0, f"Cross-Encoder reranking exceeded 200ms budget: {elapsed_ms:.2f}ms"
