"""
tests/test_load.py — FastAPI Concurrency & Latency Load Test
─────────────────────────────────────────────────────────────────────────────
Simulates N concurrent user requests against the running FastAPI service
and calculates latency percentiles (p50, p95, p99).
"""

import sys
import time
import asyncio
from pathlib import Path
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def simulate_user_request(client, query: str):
    t0 = time.perf_counter()
    response = await client.post("/find_similar", json={"question": query, "top_k": 5})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return response.status_code, elapsed_ms


async def run_load_test(total_requests: int = 50, concurrency: int = 5):
    from httpx import AsyncClient, ASGITransport
    from api.main import app
    from unittest.mock import MagicMock

    # Setup mock retriever for latency test if real model not loaded
    if getattr(app.state, "retriever", None) is None:
        mock = MagicMock()
        mock.index.ntotal = 500000
        sample_results = [
            {"question_id": 1, "question_text": "Sample duplicate question text", "score": 0.89}
        ]
        mock.search.return_value = sample_results
        mock.search_with_cache.return_value = (sample_results, False)
        app.state.retriever = mock

    transport = ASGITransport(app=app)
    queries = [
        "How do I learn Python fast?",
        "What is the difference between machine learning and deep learning?",
        "How to prepare for coding interviews?",
        "What are the best books for data science?",
        "How does gradient boosting work under the hood?",
    ]

    latencies = []
    statuses = []

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        sem = asyncio.Semaphore(concurrency)

        async def worker(q):
            async with sem:
                status, lat = await simulate_user_request(client, q)
                statuses.append(status)
                latencies.append(lat)

        tasks = [worker(queries[i % len(queries)]) for i in range(total_requests)]
        await asyncio.gather(*tasks)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print("\n" + "="*60)
    print(f" LATENCY LOAD TEST SUMMARY ({total_requests} requests, concurrency={concurrency})")
    print("="*60)
    print(f"  Success Rate : {statuses.count(200)/len(statuses):.1%}")
    print(f"  p50 Latency  : {p50:.2f} ms")
    print(f"  p95 Latency  : {p95:.2f} ms")
    print(f"  p99 Latency  : {p99:.2f} ms")
    print("="*60)

    assert statuses.count(200) == total_requests, "Some load test requests failed!"
    assert p95 < 200.0, f"p95 latency exceeded SLA of 200ms: {p95:.2f}ms"


if __name__ == "__main__":
    asyncio.run(run_load_test())
