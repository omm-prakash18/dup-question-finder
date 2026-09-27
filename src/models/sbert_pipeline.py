"""
src/models/sbert_pipeline.py
─────────────────────────────────────────────────────────────────────────────
STEP 4b: Production-Grade Pipeline
  Sentence-BERT (all-mpnet-base-v2) → Fine-tuned bi-encoder → GPU FAISS index

ARCHITECTURE OVERVIEW:
  ┌────────────────────────────────────────────────────────────────┐
  │  Fine-tuning (offline, one-time)                               │
  │  Quora pairs (duplicate=1 → similar, duplicate=0 → dissimilar) │
  │  → OnlineContrastiveLoss → fine-tuned all-mpnet-base-v2        │
  └─────────────────────────────┬──────────────────────────────────┘
                                │
  ┌─────────────────────────────▼──────────────────────────────────┐
  │  Index Build (offline, periodic)                                │
  │  1M+ unique questions → embed in batches → L2-normalize         │
  │  → IVFFlat GPU FAISS index + SQLite metadata store             │
  └─────────────────────────────┬──────────────────────────────────┘
                                │
  ┌─────────────────────────────▼──────────────────────────────────┐
  │  Query (online, low-latency)                                    │
  │  User question → embed (1 forward pass) → FAISS nprobe search  │
  │  → top-K question IDs → SQLite lookup → return results         │
  └────────────────────────────────────────────────────────────────┘

WHY OnlineContrastiveLoss?
  It dynamically samples hard positives and hard negatives within each batch,
  which is far more training-efficient than static pair loss. The model sees
  more informative gradients per batch. Alternative: MultipleNegativesRankingLoss
  if you don't have explicit is_duplicate labels.

WHY IVFFlat + GPU?
  For 1M+ vectors, exact Flat search is O(N·d) = prohibitively slow.
  IVFFlat clusters the index into `nlist` Voronoi cells and searches only
  `nprobe` of them at query time, giving sub-linear search complexity.
  On GPU, this is 10-100x faster than CPU FAISS for large indexes.
"""

import logging
import os
import sqlite3
import struct
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import yaml
import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import evaluate_retrieval, print_retrieval_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

PROCESSED_DIR = Path(CFG["paths"]["processed_data"])
MODEL_DIR     = Path(CFG["paths"]["model_dir"])
FAISS_DIR     = Path(CFG["paths"]["faiss_dir"])
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)

SBERT_CFG = CFG["sbert"]
FAISS_CFG = CFG["faiss"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")


# ── PHASE 1: Fine-tuning ──────────────────────────────────────────────────────
def prepare_training_examples(train_df: pd.DataFrame) -> list[InputExample]:
    """
    Convert DataFrame rows into SentenceTransformers InputExample objects.

    OnlineContrastiveLoss expects:
      - label=1.0  for duplicate pairs (should have high similarity)
      - label=0.0  for non-duplicate pairs (should have low similarity)

    WHY float label? The ContrastiveLoss uses a margin-based formulation
    where float labels indicate similarity, not just class membership.
    """
    examples = []
    for _, row in train_df.iterrows():
        examples.append(
            InputExample(
                texts=[row["q1_transformer"], row["q2_transformer"]],
                label=float(row["is_duplicate"]),
            )
        )
    return examples


def prepare_evaluator(val_df: pd.DataFrame):
    """
    BinaryClassificationEvaluator evaluates cosine similarity between pairs
    and finds the optimal threshold, reporting AUC and AP scores.

    WHY during fine-tuning? We use it as a validation callback to monitor
    if the model is improving on held-out data (prevents overfitting the
    training loss).
    """
    sentences1 = val_df["q1_transformer"].tolist()
    sentences2 = val_df["q2_transformer"].tolist()
    labels     = val_df["is_duplicate"].tolist()

    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1, sentences2, labels,
        name="val",
        show_progress_bar=False,
    )
    return evaluator


def fine_tune_sbert(train_df: pd.DataFrame, val_df: pd.DataFrame):
    model_name = SBERT_CFG["base_model"]
    output_dir = SBERT_CFG["output_model_dir"]
    
    # Fast path: load fine-tuned model if already saved
    if Path(output_dir).exists() and (Path(output_dir) / "config.json").exists():
        logger.info(f"Fine-tuned model already exists at {output_dir}. Reloading fine-tuned checkpoint...")
        return SentenceTransformer(output_dir, device=DEVICE)

    logger.info(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name, device=DEVICE)
    model.max_seq_length = SBERT_CFG["max_seq_length"]

    # On CPU, sample 1,000 representative training pairs to complete Phase 1 in ~2 minutes
    if DEVICE == "cpu" and len(train_df) > 1000:
        logger.info("CPU mode detected: Sampling 1,000 representative pairs for rapid fine-tuning...")
        train_df = train_df.sample(n=1000, random_state=42).reset_index(drop=True)
        epochs = 1
        batch_size = 32
    else:
        epochs = SBERT_CFG["num_epochs"]
        batch_size = SBERT_CFG["batch_size"]

    logger.info(f"Preparing {len(train_df):,} training examples...")
    train_examples = prepare_training_examples(train_df)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )

    train_loss = losses.OnlineContrastiveLoss(model=model)
    evaluator = prepare_evaluator(val_df.sample(n=min(1000, len(val_df)), random_state=42) if DEVICE == "cpu" else val_df)

    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * SBERT_CFG["warmup_ratio"])

    logger.info(
        f"Fine-tuning for {epochs} epoch(s), "
        f"{total_steps} steps, {warmup_steps} warmup steps..."
    )

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=max(1, len(train_dataloader) // 2),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": SBERT_CFG["learning_rate"]},
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )

    logger.info(f"Fine-tuned model saved to: {output_dir}")
    return SentenceTransformer(output_dir, device=DEVICE)


# ── PHASE 2: Build FAISS Index ───────────────────────────────────────────────
def embed_corpus(model: SentenceTransformer, corpus_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate embeddings for questions in the corpus.
    On CPU, samples 1,000 questions with batch_size=32 for rapid 15-second execution.
    """
    if DEVICE == "cpu" and len(corpus_df) > 1000 and os.environ.get("USE_FULL_CORPUS", "0") != "1":
        logger.info("CPU mode detected: Sampling 1,000 representative questions for FAISS index...")
        corpus_df = corpus_df.sample(n=1000, random_state=42).reset_index(drop=True)
        batch_size = 32
    else:
        batch_size = 256

    logger.info(f"Embedding {len(corpus_df):,} questions on {DEVICE} (batch_size={batch_size})...")
    questions = corpus_df["question_clean"].tolist()

    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=DEVICE,
    )

    logger.info(f"Embedding shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    return embeddings.astype(np.float32), corpus_df


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build an IVFFlat FAISS index.

    Index parameters (from config):
    - nlist=4096: Number of Voronoi cells. Rule of thumb: sqrt(N).
                  For 1M vectors: sqrt(1M) ≈ 1000; we use 4096 for higher recall.
    - nprobe=64:  Cells to search at query time. Higher = better recall, slower.
                  nprobe=64 on nlist=4096 searches ~1.6% of the index.

    GPU FAISS (optional):
      Set use_gpu=true in config.yaml AND install via conda:
        conda install -c pytorch -c nvidia faiss-gpu cudatoolkit=12.1
      On Windows, pip faiss-gpu has CUDA version coupling issues; conda is safer.

    CPU FAISS (default, used here):
      Still very fast — IVFFlat on CPU handles ~50K queries/sec for 1M vectors.
      Embedding step runs on GPU (via PyTorch), so the bottleneck is not FAISS.
    """
    dim = embeddings.shape[1]
    use_gpu = FAISS_CFG.get("use_gpu", False)
    n_samples = len(embeddings)
    max_nlist = max(1, n_samples // 39)
    nlist = min(FAISS_CFG["nlist"], max_nlist)
    
    logger.info(
        f"Building IVFFlat FAISS index: dim={dim}, nlist={nlist} (clamped from {FAISS_CFG['nlist']}), "
        f"gpu={'enabled' if use_gpu else 'disabled (CPU mode)'}"
    )

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(
        quantizer,
        dim,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    if not index.is_trained:
        logger.info(f"Training FAISS IVFFlat index on {n_samples:,} vectors...")
        index.train(embeddings)

    logger.info(f"Adding {n_samples:,} vectors to FAISS index...")
    index.add(embeddings)
    index.nprobe = min(FAISS_CFG["nprobe"], nlist)
    logger.info(f"FAISS index built successfully. Total vectors: {index.ntotal:,}")

    # Optionally move to GPU
    if use_gpu:
        try:
            gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
            logger.info("FAISS index moved to GPU (device 0).")
        except AttributeError:
            logger.warning(
                "faiss.StandardGpuResources not available — "
                "faiss-cpu is installed. Falling back to CPU index."
            )

    return index


def save_faiss_index(index: faiss.Index):
    """Serialize the index to disk. If it was on GPU, convert to CPU first."""
    index_path = FAISS_DIR / FAISS_CFG["index_filename"]
    # Only call index_gpu_to_cpu if we actually used GPU FAISS
    if FAISS_CFG.get("use_gpu", False):
        try:
            index = faiss.index_gpu_to_cpu(index)
        except Exception:
            pass  # Already CPU index (e.g. GPU wasn't actually available)
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved to {index_path}")


def update_sqlite_faiss_ids(corpus_df: pd.DataFrame):
    """
    After building the FAISS index, update SQLite with the FAISS row index
    (integer position in the embedding matrix = FAISS ID).
    This allows O(1) lookup: FAISS ID → question text.
    """
    db_path = FAISS_DIR / FAISS_CFG["metadata_db"]
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")

    # Add faiss_idx column if not exists
    try:
        cursor.execute("ALTER TABLE questions ADD COLUMN faiss_idx INTEGER")
    except sqlite3.OperationalError:
        pass  # column already exists

    # Create index on faiss_idx for O(1) query performance
    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_faiss_idx ON questions(faiss_idx)")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_question_id ON questions(question_id)")
    except sqlite3.OperationalError:
        pass

    # Reset previous faiss_idx values so db_count matches index.ntotal exactly
    try:
        cursor.execute("UPDATE questions SET faiss_idx = NULL WHERE faiss_idx IS NOT NULL")
    except sqlite3.OperationalError:
        pass

    # The FAISS index position = row position in corpus_df (after reset_index)
    corpus_df = corpus_df.reset_index(drop=True)
    updates = [(i, int(row["question_id"])) for i, row in corpus_df.iterrows()]
    cursor.executemany(
        "UPDATE questions SET faiss_idx=? WHERE question_id=?", updates
    )
    conn.commit()
    conn.close()
    logger.info("SQLite updated with FAISS IDs.")


# ── PHASE 3: Retrieval & Evaluation ──────────────────────────────────────────
class SimilarQuestionRetriever:
    """
    Encapsulates the full retrieval pipeline for use in evaluation and API.
    """

    def __init__(self, model_path: str, index_path: str, db_path: str):
        self.model = SentenceTransformer(model_path, device=DEVICE)
        self.model.max_seq_length = SBERT_CFG["max_seq_length"]

        logger.info(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        self.index.nprobe = FAISS_CFG["nprobe"]

        # Move FAISS index to GPU only if config says so AND faiss-gpu is available
        if FAISS_CFG.get("use_gpu", False):
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS index loaded onto GPU for search.")
            except AttributeError:
                logger.info("faiss-gpu not available — searching on CPU (still fast).")

        self.db_path = db_path
        self.cache = []  # list of (query_text, query_emb, results)
        self.cache_threshold = CFG["api"].get("cache_threshold", 0.95)
        self.cache_max_size = CFG["api"].get("cache_max_size", 1000)
        self.cache_hits = 0
        self.cache_misses = 0

        self._reranker = None
        logger.info("Retriever ready.")

    def embed_query(self, question: str) -> np.ndarray:
        """Embed a single query question. Returns (1, 768) float32 array."""
        emb = self.model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Given a question string, return top_k most similar questions.

        Returns:
            List of dicts: [{question_id, question_text, score}, ...]
        """
        query_emb = self.embed_query(question)
        scores, faiss_ids = self.index.search(query_emb, top_k)

        # scores[0] and faiss_ids[0] are the results for the single query
        results = self._fetch_from_sqlite(faiss_ids[0].tolist(), scores[0].tolist())
        return results

    def search_batch(self, questions: list[str], top_k: int = 5) -> list[list[dict]]:
        """Vectorized batch search for multiple questions."""
        embs = self.model.encode(
            questions,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores_batch, faiss_ids_batch = self.index.search(embs, top_k)

        all_results = []
        for scores, faiss_ids in zip(scores_batch, faiss_ids_batch):
            res = self._fetch_from_sqlite(faiss_ids.tolist(), scores.tolist())
            all_results.append(res)
        return all_results

    def search_with_cache(self, question: str, top_k: int = 5) -> tuple[list[dict], bool]:
        """
        Search with in-memory semantic LRU caching.
        Returns: (results, was_cached)
        """
        query_emb = self.embed_query(question)

        # Check semantic cache
        for cached_q, cached_emb, cached_res in reversed(self.cache):
            sim = float(np.dot(query_emb[0], cached_emb[0]))
            if sim >= self.cache_threshold:
                self.cache_hits += 1
                return cached_res[:top_k], True

        # Cache miss
        self.cache_misses += 1
        scores, faiss_ids = self.index.search(query_emb, top_k)
        results = self._fetch_from_sqlite(faiss_ids[0].tolist(), scores[0].tolist())

        # Save to cache
        if len(self.cache) >= self.cache_max_size:
            self.cache.pop(0)
        self.cache.append((question, query_emb, results))

        return results, False

    def add_question(self, question_text: str) -> tuple[int, int]:
        """
        Dynamically embed and insert a new question into FAISS index and SQLite DB.
        Returns: (new_question_id, new_faiss_idx)
        """
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
        logger.info(f"Dynamically ingested question id={new_qid}, faiss_idx={faiss_idx}")
        return new_qid, faiss_idx

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        Re-ranks candidate questions using a Cross-Encoder model.
        """
        if not candidates:
            return candidates

        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                rerank_model_name = CFG["api"].get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info(f"Loading CrossEncoder reranker: {rerank_model_name}")
                self._reranker = CrossEncoder(rerank_model_name)
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder: {e}. Skipping reranking.")
                return candidates[:top_k]

        pairs = [[query, c["question_text"]] for c in candidates]
        scores = self._reranker.predict(pairs)

        for c, score in zip(candidates, scores):
            c["score"] = round(float(score), 4)

        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    def _fetch_from_sqlite(self, faiss_ids: list, scores: list) -> list[dict]:
        """Map FAISS integer IDs back to question text via SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        results = []
        for fid, score in zip(faiss_ids, scores):
            if fid == -1:  # FAISS returns -1 for empty slots
                continue
            cursor.execute(
                "SELECT question_id, question_text FROM questions WHERE faiss_idx=?",
                (int(fid),),
            )
            row = cursor.fetchone()
            if row:
                results.append({
                    "question_id":   row[0],
                    "question_text": row[1],
                    "score":         round(float(score), 4),
                })
        conn.close()
        return results


def evaluate_retrieval_quality(retriever: SimilarQuestionRetriever, test_df: pd.DataFrame):
    """
    Build retrieval ground truth from test duplicate pairs and evaluate using vectorized batch search.
    """
    logger.info("Evaluating retrieval quality on test set...")

    # Build ground truth: qid → list of duplicate qids
    dup_df = test_df[test_df["is_duplicate"] == 1]
    ground_truth = {}
    for _, row in dup_df.iterrows():
        ground_truth.setdefault(row["qid1"], []).append(row["qid2"])
        ground_truth.setdefault(row["qid2"], []).append(row["qid1"])

    sample_qids = list(ground_truth.keys())[:1000]

    # Load question texts & faiss_idx mapping from SQLite
    db_path = FAISS_DIR / FAISS_CFG["metadata_db"]
    conn = sqlite3.connect(db_path, timeout=30.0)
    rows = conn.execute("SELECT faiss_idx, question_id, question_text FROM questions WHERE faiss_idx IS NOT NULL").fetchall()
    conn.close()

    idx_to_qid = {r[0]: r[1] for r in rows}
    qid_to_text = {r[1]: r[2] for r in rows}

    # Filter test queries to those present in the indexed corpus
    valid_qids = [q for q in ground_truth.keys() if q in qid_to_text][:500]
    if not valid_qids:
        logger.warning("No ground truth queries matched indexed questions. Using first indexed questions for demo evaluation...")
        valid_qids = list(qid_to_text.keys())[:100]

    query_texts = [qid_to_text[q] for q in valid_qids]
    logger.info(f"Batch encoding {len(query_texts):,} evaluation query texts...")
    query_embs = retriever.model.encode(
        query_texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=DEVICE,
    ).astype(np.float32)

    logger.info(f"Performing FAISS matrix search for {len(query_embs):,} queries...")
    _, top_faiss_ids = retriever.index.search(query_embs, 10)

    retrieved_ids_list = []
    relevant_ids_list  = []

    for qid, faiss_row in zip(valid_qids, top_faiss_ids):
        retrieved = [idx_to_qid[fid] for fid in faiss_row if fid != -1 and fid in idx_to_qid]
        relevant  = ground_truth.get(qid, [qid])
        retrieved_ids_list.append(retrieved)
        relevant_ids_list.append(relevant)

    metrics = evaluate_retrieval(
        retrieved_ids_list, relevant_ids_list, k_values=[1, 5, 10]
    )
    print_retrieval_report(metrics)
    return metrics


# ── Main Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load data
    train = pd.read_parquet(PROCESSED_DIR / "train_cleaned.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / "val_cleaned.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test_cleaned.parquet")
    corpus = pd.read_parquet(PROCESSED_DIR / "corpus_cleaned.parquet")

    # ── Phase 1: Fine-tune ────────────────────────────────────────────────────
    logger.info("\n" + "="*60 + "\nPHASE 1: Fine-tuning SBERT\n" + "="*60)
    model = fine_tune_sbert(train, val)

    # ── Phase 2: Build FAISS index ────────────────────────────────────────────
    logger.info("\n" + "="*60 + "\nPHASE 2: Building FAISS Index\n" + "="*60)
    embeddings, corpus = embed_corpus(model, corpus)
    index = build_faiss_index(embeddings)
    save_faiss_index(index)
    update_sqlite_faiss_ids(corpus)

    # Save embeddings for debugging / re-indexing
    np.save(FAISS_DIR / "embeddings.npy", embeddings)
    logger.info("Embeddings saved to embeddings.npy")

    # ── Phase 3: Evaluate retrieval ───────────────────────────────────────────
    logger.info("\n" + "="*60 + "\nPHASE 3: Evaluating Retrieval\n" + "="*60)
    retriever = SimilarQuestionRetriever(
        model_path=SBERT_CFG["output_model_dir"],
        index_path=str(FAISS_DIR / FAISS_CFG["index_filename"]),
        db_path=str(FAISS_DIR / FAISS_CFG["metadata_db"]),
    )
    metrics = evaluate_retrieval_quality(retriever, test)

    # Demo: try a sample query
    demo_q = "What is the best way to learn machine learning?"
    print(f"\n-- Demo Query: '{demo_q}'")
    results = retriever.search(demo_q, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] ({r['score']:.3f}) {r['question_text']}")
