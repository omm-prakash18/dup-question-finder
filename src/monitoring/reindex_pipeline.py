"""
src/monitoring/reindex_pipeline.py
─────────────────────────────────────────────────────────────────────────────
STEP 5: Production Monitoring & Periodic Re-indexing

WHAT TO MONITOR IN PRODUCTION
─────────────────────────────
1. LATENCY
   - p50/p95/p99 embedding latency (target: <50ms per query on GPU)
   - p50/p95/p99 FAISS search latency (target: <10ms for 1M IVF index)
   - End-to-end API latency via /find_similar (target: <100ms)
   → Alert if p99 > 200ms

2. EMBEDDING DRIFT
   - Compute mean cosine similarity between new incoming questions and
     the corpus centroid monthly.
   - If it drops below a threshold (e.g., <0.7 of baseline), the corpus
     has drifted semantically → re-index required.
   → Symptom: users see irrelevant results for new-style questions.

3. RETRIEVAL QUALITY (online)
   - Click-through rate (CTR) on returned suggestions:
     Frontend logs which suggested question the user clicked.
     Low CTR (<15%) → model returning irrelevant results.
   - User dismissal rate: user explicitly says "none of these are similar"

4. INDEX GROWTH
   - Monitor corpus size monthly.
   - Trigger full re-index when corpus grows >10% since last index build.
   - Or: incremental add (index.add(new_embeddings)) for small additions,
     then rebuild centroids quarterly.

5. MODEL STALENESS
   - New duplicate patterns may emerge (new slang, new topics).
   - Schedule quarterly fine-tuning on newly labeled pairs.
   - Use active learning: surface high-confidence model predictions to
     human raters for labeling.

RE-TRAINING STRATEGY
─────────────────────
When to re-train:
  - Monthly: collect new user-labeled duplicate pairs
  - Quarterly: full fine-tuning run with accumulated data
  - On drift alert: emergency re-train with recent data window

This script implements:
  A. Drift detection (monthly job)
  B. Incremental index update (add new questions without full rebuild)
  C. Full re-index pipeline (quarterly or on drift alert)
"""

import logging
import sqlite3
import argparse
import time
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import faiss
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

FAISS_DIR     = Path(CFG["paths"]["faiss_dir"])
PROCESSED_DIR = Path(CFG["paths"]["processed_data"])
SBERT_CFG     = CFG["sbert"]
FAISS_CFG     = CFG["faiss"]
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ── A. Embedding Drift Detection ──────────────────────────────────────────────
def compute_corpus_centroid(embeddings_path: Path) -> np.ndarray:
    """Load saved embeddings and compute the mean vector (corpus centroid)."""
    embeddings = np.load(embeddings_path)
    centroid = embeddings.mean(axis=0)
    # L2-normalize so we can use cosine distance on the centroid too
    centroid /= np.linalg.norm(centroid)
    return centroid


def detect_drift(new_questions: list[str], baseline_centroid: np.ndarray,
                 model: SentenceTransformer, threshold: float = 0.7) -> dict:
    """
    Compute mean cosine similarity of new questions to the corpus centroid.
    If it drops below `threshold` × baseline, flag drift.

    Args:
        new_questions:     Recently ingested questions (past 30 days)
        baseline_centroid: Saved centroid from last index build
        threshold:         Alert if similarity drops below this fraction of 1.0

    Returns:
        dict with mean_similarity, drift_detected, n_new_questions
    """
    logger.info(f"Checking embedding drift on {len(new_questions)} new questions...")
    new_embs = model.encode(
        new_questions, batch_size=256,
        normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=True,
    )
    # Cosine similarity to centroid (dot product on L2-normalized vectors)
    similarities = (new_embs @ baseline_centroid).tolist()
    mean_sim = float(np.mean(similarities))

    drift_detected = mean_sim < threshold
    if drift_detected:
        logger.warning(
            f"DRIFT DETECTED: mean cosine similarity to corpus centroid = {mean_sim:.3f} "
            f"(below threshold {threshold}). Consider re-indexing."
        )
    else:
        logger.info(f"No drift: mean cosine similarity = {mean_sim:.3f} ≥ {threshold}")

    return {
        "mean_similarity":   round(mean_sim, 4),
        "drift_detected":    drift_detected,
        "threshold":         threshold,
        "n_new_questions":   len(new_questions),
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ── B. Incremental Index Update (add new questions) ───────────────────────────
def incremental_add(new_df: pd.DataFrame, model: SentenceTransformer):
    """
    Add new questions to the existing FAISS index WITHOUT rebuilding from scratch.

    WHY this works for IVFFlat:
      IVF cluster centroids are already trained. New vectors are assigned to the
      nearest centroid and appended. This is O(new_N × nlist) not O(full_N × d).

    LIMITATION:
      After many incremental adds, centroid quality degrades (new data changes
      the optimal cluster boundaries). Do a full rebuild quarterly.

    Args:
        new_df: DataFrame with columns [question_id, question_text, question_clean]
    """
    index_path = FAISS_DIR / FAISS_CFG["index_filename"]
    db_path    = FAISS_DIR / FAISS_CFG["metadata_db"]

    logger.info(f"Loading existing FAISS index ({index_path})...")
    index = faiss.read_index(str(index_path))
    existing_size = index.ntotal
    index.nprobe = FAISS_CFG["nprobe"]

    logger.info(f"Embedding {len(new_df):,} new questions...")
    new_embs = model.encode(
        new_df["question_clean"].tolist(),
        batch_size=256, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True, device=DEVICE,
    ).astype(np.float32)

    # Append to index — new FAISS IDs = existing_size, existing_size+1, ...
    index.add(new_embs)
    logger.info(f"Index size: {existing_size:,} → {index.ntotal:,} (+{len(new_df):,})")

    # Save updated index
    faiss.write_index(index, str(index_path))

    # Update SQLite with new FAISS IDs
    conn = sqlite3.connect(db_path)
    updates = [
        (existing_size + i, row["question_id"])
        for i, row in enumerate(new_df.to_dict("records"))
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO questions (question_id, question_text, faiss_idx, source) "
        "VALUES (?, ?, ?, ?)",
        [
            (row["question_id"], row["question_text"], existing_size + i, "new_batch")
            for i, row in enumerate(new_df.to_dict("records"))
        ]
    )
    conn.commit()
    conn.close()
    logger.info("SQLite metadata updated with new FAISS IDs.")


# ── C. Full Re-index Pipeline ─────────────────────────────────────────────────
def full_reindex(model: SentenceTransformer):
    """
    Full rebuild of the FAISS index from scratch.
    Run quarterly or when drift is detected.

    This is identical to the index build in sbert_pipeline.py but modularized
    here for use as a standalone cron job.
    """
    from src.models.sbert_pipeline import (
        embed_corpus, build_faiss_index, save_faiss_index, update_sqlite_faiss_ids
    )

    logger.info("Starting full re-index...")
    corpus = pd.read_parquet(PROCESSED_DIR / "corpus_cleaned.parquet")

    # In production, corpus may have grown — reload from SQLite
    db_path = FAISS_DIR / FAISS_CFG["metadata_db"]
    conn = sqlite3.connect(db_path)
    corpus_from_db = pd.read_sql(
        "SELECT question_id, question_text FROM questions", conn
    )
    conn.close()
    corpus_from_db["question_clean"] = corpus_from_db["question_text"]
    corpus = corpus_from_db

    embeddings = embed_corpus(model, corpus)
    np.save(FAISS_DIR / "embeddings.npy", embeddings)

    index = build_faiss_index(embeddings)
    save_faiss_index(index)
    update_sqlite_faiss_ids(corpus)

    logger.info("Full re-index complete.")


# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production monitoring & re-indexing")
    parser.add_argument(
        "--mode",
        choices=["drift", "incremental", "full-reindex"],
        required=True,
        help=(
            "drift: detect embedding drift on new questions\n"
            "incremental: add new questions to existing index\n"
            "full-reindex: rebuild entire FAISS index from scratch"
        ),
    )
    parser.add_argument("--new-data", type=str, default=None,
                        help="Path to CSV/Parquet of new questions for drift/incremental modes")
    args = parser.parse_args()

    model = SentenceTransformer(SBERT_CFG["output_model_dir"], device=DEVICE)

    if args.mode == "drift":
        if not args.new_data:
            raise ValueError("--new-data required for drift mode")
        new_df = pd.read_csv(args.new_data) if args.new_data.endswith(".csv") else \
                 pd.read_parquet(args.new_data)
        centroid = compute_corpus_centroid(FAISS_DIR / "embeddings.npy")
        result = detect_drift(new_df["question_text"].tolist(), centroid, model)
        print(result)

    elif args.mode == "incremental":
        if not args.new_data:
            raise ValueError("--new-data required for incremental mode")
        new_df = pd.read_csv(args.new_data) if args.new_data.endswith(".csv") else \
                 pd.read_parquet(args.new_data)
        incremental_add(new_df, model)

    elif args.mode == "full-reindex":
        full_reindex(model)
