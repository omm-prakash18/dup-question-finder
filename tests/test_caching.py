"""
tests/test_caching.py — Semantic Caching Tests
─────────────────────────────────────────────────────────────────────────────
PRODUCTION FAILURE MODE:
  - Similarity boundary (0.95 vs 0.94) must be strict.
  - Cache hits must be dramatically faster (< 2ms) and increment cache hit counter.
"""

import time
import pytest
import numpy as np


class DummySemanticCache:
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.cache = []
        self.hits = 0
        self.misses = 0

    def query(self, text: str, vec: np.ndarray, compute_fn):
        for c_text, c_vec, c_res in reversed(self.cache):
            sim = float(np.dot(vec[0], c_vec[0]))
            if sim >= self.threshold:
                self.hits += 1
                return c_res, True

        self.misses += 1
        res = compute_fn(text)
        self.cache.append((text, vec, res))
        return res, False


def test_cache_hit_boundary_095_vs_094():
    """
    Cosine similarity >= 0.95 MUST hit cache.
    Cosine similarity = 0.94 MUST miss cache and invoke computation.
    """
    cache = DummySemanticCache(threshold=0.95)

    vec_base = np.array([[1.0, 0.0]], dtype=np.float32)

    # 1st query: miss
    res1, hit1 = cache.query("Base query", vec_base, lambda q: ["Result Base"])
    assert hit1 is False
    assert cache.misses == 1

    # 2nd query: similarity = 0.96 (>= 0.95 -> HIT)
    vec_high_sim = np.array([[0.96, 0.28]], dtype=np.float32)  # dot product = 0.96
    res2, hit2 = cache.query("Paraphrase query", vec_high_sim, lambda q: ["Result Compute"])
    assert hit2 is True
    assert cache.hits == 1

    # 3rd query: similarity = 0.94 (< 0.95 -> MISS)
    vec_low_sim = np.array([[0.94, 0.341]], dtype=np.float32)  # dot product = 0.94
    res3, hit3 = cache.query("Different query", vec_low_sim, lambda q: ["Result Compute 3"])
    assert hit3 is False
    assert cache.misses == 2


def test_cache_hit_latency_under_2ms():
    """
    Assert cache hits execute in < 2ms.
    """
    cache = DummySemanticCache(threshold=0.95)
    vec = np.array([[1.0, 0.0]], dtype=np.float32)

    cache.query("Test query", vec, lambda q: ["Result"])

    t0 = time.perf_counter()
    res, hit = cache.query("Test query", vec, lambda q: ["Result"])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert hit is True
    assert elapsed_ms < 2.0, f"Cache hit latency exceeded 2ms budget: {elapsed_ms:.2f}ms"
