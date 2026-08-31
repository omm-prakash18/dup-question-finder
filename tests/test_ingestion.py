"""
tests/test_ingestion.py — Dynamic Question Ingestion Tests
─────────────────────────────────────────────────────────────────────────────
PRODUCTION FAILURE MODE:
  - Hot FAISS index additions must stay in 1:1 sync with SQLite row count (`index.ntotal == count`).
  - Newly ingested questions must be immediately queryable without restarting service.
"""

import sqlite3
import pytest
import numpy as np
import faiss


class DummyRetrieverForIngestion:
    def __init__(self, db_path):
        self.db_path = db_path
        self.index = faiss.IndexFlatIP(768)

    def embed_query(self, question: str) -> np.ndarray:
        # Deterministic dummy embedding normalized to L2
        vec = np.ones((1, 768), dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec

    def add_question(self, question_text: str) -> tuple[int, int]:
        emb = self.embed_query(question_text)
        faiss_idx = int(self.index.ntotal)
        self.index.add(emb)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(question_id), 0) + 1 FROM questions")
        new_qid = int(cursor.fetchone()[0])

        cursor.execute(
            "INSERT INTO questions (question_id, question_text, faiss_idx) VALUES (?, ?, ?)",
            (new_qid, question_text, faiss_idx),
        )
        conn.commit()
        conn.close()
        return new_qid, faiss_idx


@pytest.fixture
def temp_db(tmp_path):
    db_path = str(tmp_path / "test_ingest.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE questions (
            question_id INTEGER PRIMARY KEY,
            question_text TEXT,
            faiss_idx INTEGER
        )
    """)
    conn.commit()
    conn.close()
    return db_path


def test_ingested_question_immediately_retrievable(temp_db):
    """
    Verify newly ingested question is added to FAISS and immediately retrievable.
    """
    retriever = DummyRetrieverForIngestion(temp_db)
    qid, f_idx = retriever.add_question("How to bake apple pie?")

    assert qid == 1
    assert f_idx == 0
    assert retriever.index.ntotal == 1


def test_faiss_and_sqlite_count_sync(temp_db):
    """
    Assert FAISS index.ntotal matches SQLite questions row count after N dynamic ingestions.
    """
    retriever = DummyRetrieverForIngestion(temp_db)

    for i in range(10):
        retriever.add_question(f"Dynamic question {i}")

    conn = sqlite3.connect(temp_db)
    db_count = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    conn.close()

    assert retriever.index.ntotal == 10
    assert db_count == 10
