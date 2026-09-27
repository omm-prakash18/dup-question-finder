# Repository Gotchas & Known Failure Modes

## 1. Lifespan Mock Injections vs Model Files
- **Gotcha**: When running FastAPI tests via `TestClient(app)`, `TestClient` executes the `lifespan` context manager.
- **Rule**: Never allow lifespan to unconditionally instantiate `SimilarQuestionRetriever("data/models/sbert_finetuned")` if the local path doesn't exist, as SentenceTransformer will attempt to treat it as a remote Hugging Face repo ID and hang on network timeouts. Always check `app.state.retriever` first.

## 2. FAISS Cosine Metric & L2 Normalization
- **Gotcha**: FAISS uses `METRIC_INNER_PRODUCT` for vector distance.
- **Rule**: All query and corpus embeddings MUST be strictly L2-normalized (`normalize_embeddings=True` or `vec / norm(vec)`). Unnormalized vectors will produce invalid scores outside `[-1.0, 1.0]`.

## 3. Dynamic Index Ingestion Sync
- **Gotcha**: Dynamically inserting a new question requires updating BOTH the in-memory FAISS index and the SQLite metadata database.
- **Rule**: Maintain strict 1:1 invariant: `index.ntotal == SELECT COUNT(*) FROM questions`.

## 4. Cross-Encoder Tuple Unpacking
- **Gotcha**: `retriever.search_with_cache` returns `(results, was_cached)` tuple, whereas `retriever.search` returns `results` list.
- **Rule**: When mocking retriever in tests, configure both methods with their expected return signatures.
