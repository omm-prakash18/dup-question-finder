"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
STEP 5: FastAPI Service — Similar Question Finder

Endpoints:
  GET  /health           — liveness/readiness check
  POST /find_similar     — core retrieval endpoint
  GET  /docs             — auto-generated Swagger UI

Design decisions:
  - Model and FAISS index loaded ONCE at startup via lifespan context manager.
    This avoids re-loading the 420MB model on every request (which would add
    ~5s latency per call).
  - The retriever is stored in `app.state` for thread-safe access.
  - Request logging middleware records latency for every request to enable
    p99 latency monitoring in production.

Production monitoring advice (inline below):
  - LOG: request latency, top_k, score_threshold, query length
  - MONITOR: embedding drift (monthly cosine similarity to centroid),
             click-through rate on returned suggestions (requires frontend signal),
             index size growth (trigger re-indexing when corpus grows >10%)
"""

import time
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Make src importable inside Docker and local dev
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.sbert_pipeline import SimilarQuestionRetriever
from api.schemas import (
    SimilarQuestionRequest,
    SimilarQuestionResponse,
    SimilarQuestion,
    HealthResponse,
    BatchSimilarQuestionRequest,
    BatchSimilarQuestionResponse,
    IngestQuestionRequest,
    IngestQuestionResponse,
    UserFeedbackRequest,
    UserFeedbackResponse,
    MetricsResponse,
)

# ── Config ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MODEL_PATH = CFG["sbert"]["output_model_dir"]
INDEX_PATH = str(Path(CFG["paths"]["faiss_dir"]) / CFG["faiss"]["index_filename"])
DB_PATH    = str(Path(CFG["paths"]["faiss_dir"]) / CFG["faiss"]["metadata_db"])
FEEDBACK_DB = str(Path(__file__).resolve().parent.parent / CFG["api"].get("feedback_db", "data/faiss/feedback.db"))


# ── Lifespan: load model once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler (replaces deprecated on_event startup/shutdown).
    Loads the retriever at startup; releases resources at shutdown.
    """
    logger.info("Loading SBERT model and FAISS index...")
    app.state.start_time = time.time()
    if not hasattr(app.state, "request_count"):
        app.state.request_count = 0

    if getattr(app.state, "retriever", None) is None:
        if Path(MODEL_PATH).exists() and Path(INDEX_PATH).exists() and Path(DB_PATH).exists():
            try:
                retriever = SimilarQuestionRetriever(
                    model_path=MODEL_PATH,
                    index_path=INDEX_PATH,
                    db_path=DB_PATH,
                )
                app.state.retriever = retriever
                logger.info("Service ready.")
            except Exception as e:
                logger.error(f"Failed to load model/index: {e}")
                app.state.retriever = None
        else:
            logger.info("Model or index files not found on disk. Service running without preloaded retriever.")
            app.state.retriever = None

    # Initialize feedback table in SQLite
    try:
        import sqlite3
        Path(FEEDBACK_DB).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(FEEDBACK_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                retrieved_question_id INTEGER NOT NULL,
                is_relevant INTEGER NOT NULL,
                user_rating INTEGER
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not initialize feedback DB: {e}")

    yield  # Application runs here

    logger.info("Shutting down service.")


# ── App Instance ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Similar Question Finder API",
    description=(
        "Semantic duplicate question retrieval using Sentence-BERT + FAISS. "
        "Given a question, returns the top-K most semantically similar questions "
        "from a 1M+ corpus, with cosine similarity scores."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middleware: request latency logging ───────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    if hasattr(app.state, "request_count"):
        app.state.request_count += 1

    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} ({elapsed_ms:.1f}ms)"
    )
    return response


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    retriever = getattr(app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check startup logs.",
        )

    import torch
    return HealthResponse(
        status="ok",
        model_loaded=True,
        index_size=int(retriever.index.ntotal),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@app.post("/find_similar", response_model=SimilarQuestionResponse, tags=["Retrieval"])
async def find_similar(body: SimilarQuestionRequest):
    retriever = getattr(app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please try again later.",
        )

    t0 = time.perf_counter()

    try:
        # Check semantic cache if available
        if hasattr(retriever, "search_with_cache"):
            raw_results, was_cached = retriever.search_with_cache(body.question, top_k=body.top_k * 2)
        else:
            raw_results = retriever.search(body.question, top_k=body.top_k * 2)
    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    # Rerank if enabled in config
    if CFG["api"].get("rerank_enabled", False) and hasattr(retriever, "rerank"):
        raw_results = retriever.rerank(body.question, raw_results, top_k=body.top_k * 2)

    # Filter by score threshold and truncate to top_k
    filtered = [
        SimilarQuestion(**r)
        for r in raw_results
        if r["score"] >= body.score_threshold
    ][:body.top_k]

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"find_similar | "
        f"query_len={len(body.question)} | "
        f"top_k={body.top_k} | "
        f"threshold={body.score_threshold} | "
        f"results={len(filtered)} | "
        f"latency={latency_ms:.1f}ms"
    )

    return SimilarQuestionResponse(
        query=body.question,
        results=filtered,
        result_count=len(filtered),
        latency_ms=round(latency_ms, 2),
    )


@app.post("/find_similar/batch", response_model=BatchSimilarQuestionResponse, tags=["Retrieval"])
async def find_similar_batch(body: BatchSimilarQuestionRequest):
    """
    Vectorized batch search for multiple query questions.
    """
    retriever = getattr(app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="Model not available.")

    t0 = time.perf_counter()

    try:
        if hasattr(retriever, "search_batch"):
            batch_results = retriever.search_batch(body.questions, top_k=body.top_k * 2)
        else:
            batch_results = [retriever.search(q, top_k=body.top_k * 2) for q in body.questions]
    except Exception as e:
        logger.error(f"Batch retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch retrieval failed: {str(e)}")

    responses = []
    for q, raw_res in zip(body.questions, batch_results):
        filtered = [
            SimilarQuestion(**r)
            for r in raw_res
            if r["score"] >= body.score_threshold
        ][:body.top_k]

        responses.append(
            SimilarQuestionResponse(
                query=q,
                results=filtered,
                result_count=len(filtered),
                latency_ms=round((time.perf_counter() - t0) * 1000 / len(body.questions), 2),
            )
        )

    total_latency = (time.perf_counter() - t0) * 1000
    return BatchSimilarQuestionResponse(
        query_count=len(body.questions),
        responses=responses,
        total_latency_ms=round(total_latency, 2),
    )


@app.post("/questions", response_model=IngestQuestionResponse, tags=["Management"])
async def ingest_question(body: IngestQuestionRequest):
    """
    Dynamically add a new question to the FAISS index and SQLite metadata store.
    """
    retriever = getattr(app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(status_code=503, detail="Model not available.")

    try:
        new_qid, faiss_idx = retriever.add_question(body.question_text)
        return IngestQuestionResponse(
            question_id=new_qid,
            question_text=body.question_text,
            index_size=int(retriever.index.ntotal),
            status="success",
        )
    except Exception as e:
        logger.error(f"Failed to ingest question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/feedback", response_model=UserFeedbackResponse, tags=["Feedback"])
async def submit_feedback(body: UserFeedbackRequest):
    """
    Log user relevance ratings into SQLite feedback database.
    """
    import sqlite3
    try:
        Path(FEEDBACK_DB).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(FEEDBACK_DB)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                retrieved_question_id INTEGER NOT NULL,
                is_relevant INTEGER NOT NULL,
                user_rating INTEGER
            )
        """)
        cursor.execute(
            """
            INSERT INTO feedback (query, retrieved_question_id, is_relevant, user_rating)
            VALUES (?, ?, ?, ?)
            """,
            (body.query, body.retrieved_question_id, 1 if body.is_relevant else 0, body.user_rating),
        )
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return UserFeedbackResponse(feedback_id=feedback_id, status="logged")
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feedback logging failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Get operational performance metrics.
    """
    retriever = getattr(app.state, "retriever", None)
    index_size = int(retriever.index.ntotal) if retriever else 0
    cache_hits = getattr(retriever, "cache_hits", 0) if retriever else 0
    cache_misses = getattr(retriever, "cache_misses", 0) if retriever else 0
    total_cache = cache_hits + cache_misses
    hit_rate = round(cache_hits / total_cache, 4) if total_cache > 0 else 0.0

    start_time = getattr(app.state, "start_time", time.time())
    uptime = round(time.time() - start_time, 2)
    req_count = getattr(app.state, "request_count", 0)

    return MetricsResponse(
        total_requests=req_count,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        cache_hit_rate=hit_rate,
        index_size=index_size,
        uptime_seconds=uptime,
    )


# ── Dev Server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=CFG["api"]["host"],
        port=CFG["api"]["port"],
        reload=True,    # hot-reload on code changes (dev only)
        workers=1,      # single worker in dev; use gunicorn for prod
    )
