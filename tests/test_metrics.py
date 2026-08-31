"""
tests/test_metrics.py — Prometheus & System Metrics Tests
─────────────────────────────────────────────────────────────────────────────
PRODUCTION FAILURE MODE:
  - /metrics endpoint must report accurate operational metrics (index size, uptime).
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import app
    retriever = MagicMock()
    retriever.index.ntotal = 500000
    retriever.cache_hits = 15
    retriever.cache_misses = 5
    app.state.retriever = retriever
    return TestClient(app)


def test_metrics_endpoint_response(client):
    """Verify /metrics endpoint returns operational metrics."""
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()

    assert data["index_size"] == 500000
    assert data["cache_hits"] == 15
    assert data["cache_misses"] == 5
    assert data["cache_hit_rate"] == 0.75
    assert "uptime_seconds" in data
