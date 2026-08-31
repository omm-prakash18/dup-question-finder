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


# ── Lifespan: load model once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler (replaces deprecated on_event startup/shutdown).
    Loads the retriever at startup; releases resources at shutdown.

    WHY load at startup? Models are large (~420MB). Loading them per-request
    would make the service unusably slow. Startup loading amortizes the cost
    across all requests served by the process.
    """
    logger.info("Loading SBERT model and FAISS index...")
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
        logger.error(
            "Run src/models/sbert_pipeline.py first to generate the model and index."
        )
        app.state.retriever = None

    yield  # Application runs here

    # Cleanup (e.g., close DB connections if needed)
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
    """
    Log every request with method, path, status code, and latency.

    In production: ship these logs to a monitoring system (Datadog, Grafana,
    CloudWatch) and alert if p99 latency exceeds 200ms.
    """
    start = time.perf_counter()
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
    """
    Liveness + readiness probe.

    Kubernetes/Docker Compose health checks call this endpoint.
    Returns 503 if model failed to load (so the orchestrator can restart).
    """
    retriever = app.state.retriever
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
    """
    Core retrieval endpoint.

    Given a question string, returns top-K semantically similar questions
    from the indexed corpus with cosine similarity scores.

    Example:
        POST /find_similar
        {"question": "How do I reverse a linked list?", "top_k": 5}
    """
    retriever = app.state.retriever
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please try again later.",
        )

    t0 = time.perf_counter()

    try:
        # Retrieve more than top_k initially so we can filter by threshold
        raw_results = retriever.search(body.question, top_k=body.top_k * 2)
    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    # Filter by score threshold and truncate to top_k
    filtered = [
        SimilarQuestion(**r)
        for r in raw_results
        if r["score"] >= body.score_threshold
    ][:body.top_k]

    latency_ms = (time.perf_counter() - t0) * 1000

    # PRODUCTION LOG: structured log for observability
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


# ── Dev Server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=CFG["api"]["host"],
        port=CFG["api"]["port"],
        reload=True,    # hot-reload on code changes (dev only)
        workers=1,      # single worker in dev; use gunicorn for prod
    )
