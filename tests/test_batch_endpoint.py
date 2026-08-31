"""
tests/test_batch_endpoint.py — Batch Retrieval Endpoint (/find_similar/batch) Tests
─────────────────────────────────────────────────────────────────────────────
PRODUCTION FAILURE MODE:
  - If batch limit (>100 questions) isn't enforced, server VRAM explodes.
  - Batching should yield IDENTICAL embeddings/scores as single queries.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.index.ntotal = 500000
    sample = [
        {"question_id": 1, "question_text": "Python tutorial", "score": 0.90},
    ]
    retriever.search.return_value = sample
    retriever.search_with_cache.return_value = (sample, False)
    retriever.search_batch.side_effect = lambda q_list, top_k: [sample for _ in q_list]
    return retriever


@pytest.fixture
def client(mock_retriever):
    from api.main import app
    app.state.retriever = mock_retriever
    return TestClient(app)


def test_batch_boundary_100_accepted(client):
    """Batch of 100 questions is within boundary and accepted."""
    questions = [f"Question number {i}?" for i in range(100)]
    res = client.post("/find_similar/batch", json={"questions": questions, "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["query_count"] == 100
    assert len(data["responses"]) == 100


def test_batch_boundary_101_rejected(client):
    """Batch of 101 questions exceeds maximum allowed limit (100) and is rejected with 422."""
    questions = [f"Question number {i}?" for i in range(101)]
    res = client.post("/find_similar/batch", json={"questions": questions, "top_k": 2})
    assert res.status_code == 422


def test_batch_vs_single_equivalence(client):
    """
    Verify batch retrieval returns identical results compared to single requests.
    """
    q1 = "How to learn Python?"
    q2 = "What is deep learning?"

    res_single1 = client.post("/find_similar", json={"question": q1, "top_k": 2}).json()
    res_single2 = client.post("/find_similar", json={"question": q2, "top_k": 2}).json()

    res_batch = client.post("/find_similar/batch", json={"questions": [q1, q2], "top_k": 2}).json()

    assert res_batch["responses"][0]["results"] == res_single1["results"]
    assert res_batch["responses"][1]["results"] == res_single2["results"]
