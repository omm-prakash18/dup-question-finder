"""
tests/test_feedback.py — User Feedback Collection Endpoint Tests
─────────────────────────────────────────────────────────────────────────────
PRODUCTION FAILURE MODE:
  - User relevance feedback (thumbs up/down) must persist accurately.
  - Invalid rating values (e.g. rating > 5 or rating < 1) must be rejected with 422.
"""

import sqlite3
import pytest
from pathlib import Path
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    from api.main import app
    app.state.retriever = None
    return TestClient(app)


def test_valid_feedback_submission(client):
    """Verify valid user rating submission is logged with 200 status."""
    payload = {
        "query": "How to learn Python?",
        "retrieved_question_id": 101,
        "is_relevant": True,
        "user_rating": 5,
    }
    res = client.post("/feedback", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "logged"
    assert "feedback_id" in data


def test_invalid_rating_value_rejected(client):
    """Rating value of 10 (exceeding max 5) must be rejected with 422 error."""
    payload = {
        "query": "How to learn Python?",
        "retrieved_question_id": 101,
        "is_relevant": True,
        "user_rating": 10,
    }
    res = client.post("/feedback", json=payload)
    assert res.status_code == 422
