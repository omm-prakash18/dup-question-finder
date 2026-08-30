"""
tests/test_api.py — FastAPI Endpoint Comprehensive Integration Tests
─────────────────────────────────────────────────────────────────────────────
Verifies:
  1. /health Endpoint:
     a. Returns 200 OK when model is loaded.
     b. Returns 503 Service Unavailable gracefully when model/index is missing.
  2. /find_similar Endpoint:
     a. Valid Input: Returns top-K results sorted by similarity score in descending order.
     b. Determinism: Identical queries return identical top-K scores and order.
     c. Empty / Whitespace string: Rejected with 422 Unprocessable Entity.
     d. Too Short string (<3 chars): Rejected with 422 Unprocessable Entity.
     e. Extremely Long Input (>1000 chars): Rejected with 422 Unprocessable Entity.
     f. Non-UTF8 & Special Characters: Handled gracefully without server error (500).
     g. Missing required fields / Malformed JSON: Rejected with 422.
     h. Score threshold filtering: Filters out results below requested score_threshold.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def mock_retriever():
    """Mock retriever returning deterministic results."""
    retriever = MagicMock()
    retriever.index.ntotal = 500000
    retriever.search.return_value = [
        {"question_id": 101, "question_text": "How do I learn Python programming quickly?", "score": 0.94},
        {"question_id": 102, "question_text": "Best resources for Python beginners", "score": 0.88},
        {"question_id": 103, "question_text": "What is the best way to master Python?", "score": 0.76},
        {"question_id": 104, "question_text": "How long does Python take to learn?", "score": 0.45},
    ]
    return retriever


@pytest.fixture
def client(mock_retriever):
    """Test client with mock retriever injected into app.state."""
    from api.main import app
    app.state.retriever = mock_retriever
    return TestClient(app)


class TestHealthCheckEndpoint:
    def test_health_success(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["index_size"] == 500000

    def test_health_503_graceful_failure(self):
        """When retriever fails to load, /health returns 503 rather than crashing."""
        from api.main import app
        app.state.retriever = None
        c = TestClient(app, raise_server_exceptions=False)
        res = c.get("/health")
        assert res.status_code == 503
        assert "Model not loaded" in res.json()["detail"]


class TestFindSimilarEndpointValidation:
    def test_valid_request(self, client):
        res = client.post("/find_similar", json={"question": "How to learn Python?", "top_k": 3})
        assert res.status_code == 200
        data = res.json()
        assert data["query"] == "How to learn Python?"
        assert data["result_count"] == 3
        assert "latency_ms" in data

    def test_results_are_sorted_descending_by_score(self, client):
        res = client.post("/find_similar", json={"question": "How to learn Python?", "top_k": 4, "score_threshold": 0.0})
        data = res.json()
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True), f"Results not sorted descending: {scores}"

    def test_determinism_check(self, client):
        """Identical queries should yield identical results."""
        res1 = client.post("/find_similar", json={"question": "How to learn Python?", "top_k": 3})
        res2 = client.post("/find_similar", json={"question": "How to learn Python?", "top_k": 3})
        assert res1.json()["results"] == res2.json()["results"]

    def test_empty_string_rejected(self, client):
        res = client.post("/find_similar", json={"question": "   ", "top_k": 5})
        assert res.status_code == 422

    def test_too_short_string_rejected(self, client):
        res = client.post("/find_similar", json={"question": "ab", "top_k": 5})
        assert res.status_code == 422

    def test_extremely_long_input_rejected(self, client):
        long_q = "a" * 1005
        res = client.post("/find_similar", json={"question": long_q, "top_k": 5})
        assert res.status_code == 422

    def test_special_characters_and_emojis(self, client):
        special_q = "How to fix CUDA error? 🐍 🚀 &lt;script&gt;alert(1)&lt;/script&gt;"
        res = client.post("/find_similar", json={"question": special_q, "top_k": 2})
        assert res.status_code == 200

    def test_malformed_json(self, client):
        res = client.post("/find_similar", content="{bad_json:", headers={"Content-Type": "application/json"})
        assert res.status_code == 422

    def test_missing_required_field(self, client):
        res = client.post("/find_similar", json={"top_k": 5})  # missing "question"
        assert res.status_code == 422

    def test_score_threshold_filtering(self, client):
        res = client.post(
            "/find_similar",
            json={"question": "How to learn Python?", "top_k": 5, "score_threshold": 0.80}
        )
        assert res.status_code == 200
        results = res.json()["results"]
        # Only scores >= 0.80 should pass filter
        assert all(r["score"] >= 0.80 for r in results)
