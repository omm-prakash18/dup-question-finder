# Architecture Memory: dup-question-finder

## Overview
`dup-question-finder` is a high-performance NLP pipeline & FastAPI microservice for semantic duplicate question detection and sub-millisecond retrieval.

## System Topology & Data Flow

```
[Client / Query] ──► [FastAPI REST Service (/find_similar, /find_similar/batch, /questions)]
                           │
                           ├──► [Semantic LRU Cache (Threshold >= 0.95)]
                           │         │ (Hit: <2ms return)
                           │         ▼ (Miss)
                           ├──► [Sentence-BERT Bi-Encoder (all-mpnet-base-v2, 768-dim)]
                           │         │
                           │         ▼
                           ├──► [FAISS IVFFlat Vector Index (Cosine / Inner Product)]
                           │         │ (Top-2K Candidate IDs)
                           │         ▼
                           ├──► [Cross-Encoder Re-ranker (Optional MS-MARCO MiniLM)]
                           │         │
                           │         ▼
                           └──► [SQLite Metadata DB (O(1) Text Mapping & Feedback)]
```

## Key Components & Paths
- **FastAPI Service**: `api/main.py` & `api/schemas.py`
- **Model Pipeline & Retriever**: `src/models/sbert_pipeline.py`
- **Data Loaders & Cleaning**: `src/data/download_data.py` & `src/data/preprocess.py`
- **Classical Baseline (XGBoost + TF-IDF)**: `src/models/baseline.py` & `src/features/feature_engineering.py`
- **Hybrid Search (BM25 + RRF)**: `src/models/hybrid_search.py`
- **Continuous Monitoring & Reindexing**: `src/monitoring/reindex_pipeline.py`
- **Test Suites (56 Tests)**: `tests/`
