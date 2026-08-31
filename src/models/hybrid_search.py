"""
src/models/hybrid_search.py
─────────────────────────────────────────────────────────────────────────────
HYBRID SEARCH & RECIPROCAL RANK FUSION (RRF) MODULE

Combines:
  1. Dense Retrieval (SBERT + FAISS): captures semantic meaning & paraphrases
  2. Sparse Retrieval (BM25): captures exact keywords, error codes (0x80070005), code identifiers (pd.read_parquet)
  3. Reciprocal Rank Fusion (RRF): fuses rank lists without requiring score scale normalization

RRF Formula:
  RRF_score(d) = sum_{m in retrievers} (1 / (k + rank_m(d)))
  where k = 60 (standard RRF smoothing constant)
"""

import math
import re
from collections import defaultdict
from typing import Callable, Optional


def default_bm25_tokenizer(text: str) -> list[str]:
    """
    Standard tokenizer reference used at both index-building and query time.
    Keeps alphanumeric characters and special identifiers like error codes.
    """
    if not isinstance(text, str):
        return []
    return re.findall(r"\w+|0x[0-9a-fA-F]+", text.lower())


class SimpleBM25:
    """
    Pure Python Okapi BM25 retriever for exact keyword matching.
    Does not require external C dependencies.
    """

    def __init__(self, corpus: list[dict], tokenizer: Optional[Callable[[str], list[str]]] = None, k1: float = 1.5, b: float = 0.75):
        self.tokenizer = tokenizer or default_bm25_tokenizer
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = []
        self.doc_freqs = []
        self.idf = {}
        self.avgdl = 0.0

        self._build_index()

    def _build_index(self):
        if not self.corpus:
            return

        df = defaultdict(int)
        total_len = 0

        for doc in self.corpus:
            tokens = self.tokenizer(doc["question_text"])
            self.doc_len.append(len(tokens))
            total_len += len(tokens)

            freqs = defaultdict(int)
            for t in tokens:
                freqs[t] += 1
            self.doc_freqs.append(freqs)

            for t in set(tokens):
                df[t] += 1

        self.avgdl = total_len / len(self.corpus) if self.corpus else 0.0
        num_docs = len(self.corpus)

        for term, freq in df.items():
            self.idf[term] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.corpus:
            return []

        tokens = self.tokenizer(query)
        if not tokens:
            return []

        scores = [0.0] * len(self.corpus)

        for t in tokens:
            if t not in self.idf:
                continue
            idf_val = self.idf[t]
            for i, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(t, 0)
                if tf > 0:
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / self.avgdl))
                    scores[i] += idf_val * (numerator / denominator)

        # Sort documents by score descending
        indexed_scores = [(i, score) for i, score in enumerate(scores) if score > 0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            doc = dict(self.corpus[idx])
            doc["score"] = round(score, 4)
            results.append(doc)

        return results


def reciprocal_rank_fusion(
    rank_lists: list[list[dict]],
    k: int = 60,
    id_key: str = "question_id",
    top_k: int = 10
) -> list[dict]:
    """
    Fuses multiple ranked lists using Reciprocal Rank Fusion.

    RRF_score(d) = sum_{l in rank_lists} (1 / (k + rank(d)))
    """
    rrf_scores = defaultdict(float)
    doc_map = {}

    for rank_list in rank_lists:
        for rank, doc in enumerate(rank_list, start=1):
            doc_id = doc[id_key]
            doc_map[doc_id] = doc
            rrf_scores[doc_id] += 1.0 / (k + rank)

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)

    fused_results = []
    for doc_id in sorted_doc_ids[:top_k]:
        doc_copy = dict(doc_map[doc_id])
        doc_copy["score"] = round(rrf_scores[doc_id], 6)
        fused_results.append(doc_copy)

    return fused_results
