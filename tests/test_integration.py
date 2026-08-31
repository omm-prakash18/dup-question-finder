"""
tests/test_integration.py — Full End-to-End Integration & Chaos Tests
─────────────────────────────────────────────────────────────────────────────
PRODUCTION FAILURE MODE:
  - System must function seamlessly across component interactions:
    Ingest Question -> Retrieve via Batch -> Submit Feedback -> Check Metrics.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import app

    retriever = MagicMock()
    retriever.index.ntotal = 1000
    retriever.cache_hits = 0
    retriever.cache_misses = 1

    sample_item = [{"question_id": 1001, "question_text": "How to configure Docker?", "score": 0.92}]
    retriever.search_batch.return_value = [sample_item]
    retriever.search_with_cache.return_value = (sample_item, False)
    retriever.add_question.return_value = (1001, 1000)

    app.state.retriever = retriever
    return TestClient(app)


def test_end_to_end_flow(client):
    """
    Full End-to-End Flow:
      1. Ingest new question via POST /questions
      2. Query for question via POST /find_similar/batch
      3. Log relevance feedback via POST /feedback
      4. Verify operational metrics via GET /metrics
    """
    # 1. Ingest
    res_ingest = client.post("/questions", json={"question_text": "How to configure Docker?"})
    assert res_ingest.status_code == 200
    qid = res_ingest.json()["question_id"]

    # 2. Batch Query
    res_batch = client.post("/find_similar/batch", json={"questions": ["How to configure Docker?"], "top_k": 1})
    assert res_batch.status_code == 200
    assert len(res_batch.json()["responses"]) == 1

    # 3. Feedback
    res_fb = client.post(
        "/feedback",
        json={"query": "How to configure Docker?", "retrieved_question_id": qid, "is_relevant": True, "user_rating": 5}
    )
    assert res_fb.status_code == 200
    assert res_fb.json()["status"] == "logged"

    # 4. Metrics
    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200
    assert "index_size" in res_metrics.json()
