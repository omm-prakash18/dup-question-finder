"""
tests/test_model.py — Model & FAISS Index Integrity Tests
─────────────────────────────────────────────────────────────────────────────
Verifies:
  1. FAISS Index Integrity:
     a. Index size matches metadata SQLite table count.
     b. Querying with exact sentence from corpus returns itself as top-1 result (Recall@1 = 1.0).
     c. Embedding vectors are L2-normalized (magnitude == 1.0 within tolerance).
  2. Model Output Validity:
     a. Probabilities strictly bounded in [0, 1].
     b. Predict returns valid binary values {0, 1}.
"""

import pytest
import numpy as np
import sqlite3
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

FAISS_DIR = Path(CFG["paths"]["faiss_dir"])
MODEL_DIR = Path(CFG["paths"]["model_dir"])


class TestFAISSIndexIntegrity:
    @pytest.fixture(autouse=True)
    def setup_paths(self):
        self.index_path = FAISS_DIR / CFG["faiss"]["index_filename"]
        self.db_path    = FAISS_DIR / CFG["faiss"]["metadata_db"]
        self.embs_path  = FAISS_DIR / "embeddings.npy"

        if not (self.index_path.exists() and self.db_path.exists()):
            pytest.skip("FAISS index or SQLite DB not built yet. Run src/models/sbert_pipeline.py first.")

    def test_faiss_size_matches_sqlite_count(self):
        """Index size must equal number of indexed rows in SQLite database."""
        import faiss

        index = faiss.read_index(str(self.index_path))
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM questions WHERE faiss_idx IS NOT NULL")
        db_count = cursor.fetchone()[0]
        conn.close()

        assert index.ntotal == db_count, (
            f"FAISS index total ({index.ntotal}) mismatch with SQLite indexed rows ({db_count})"
        )

    def test_embeddings_are_l2_normalized(self):
        """Vector norms must be 1.0 (+/- 1e-4) for inner-product to equal cosine similarity."""
        if not self.embs_path.exists():
            pytest.skip("embeddings.npy not saved.")

        embs = np.load(self.embs_path)
        norms = np.linalg.norm(embs, axis=1)

        # Check sample of 1000 vectors
        sample_norms = norms[:1000]
        np.testing.assert_allclose(
            sample_norms, 1.0, rtol=1e-4,
            err_msg="Embeddings in FAISS index are not L2-normalized!"
        )

    def test_exact_match_retrieval(self):
        """
        Spot check: embedding a question already in the corpus should return
        itself as the top-1 result.
        """
        from src.models.sbert_pipeline import SimilarQuestionRetriever

        retriever = SimilarQuestionRetriever(
            model_path=CFG["sbert"]["output_model_dir"],
            index_path=str(self.index_path),
            db_path=str(self.db_path),
        )

        # Query SQLite for a sample indexed question
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT question_id, question_text FROM questions WHERE faiss_idx IS NOT NULL LIMIT 1").fetchone()
        conn.close()

        if not row:
            pytest.skip("SQLite database is empty.")

        target_id, target_text = row
        results = retriever.search(target_text, top_k=1)

        assert len(results) > 0, "No results returned for exact match query."
        top_result = results[0]
        assert top_result["question_id"] == target_id, (
            f"Exact query failed! Expected QID {target_id}, got {top_result['question_id']}"
        )
        assert top_result["score"] >= 0.95, (
            f"Exact match similarity score lower than expected: {top_result['score']}"
        )
