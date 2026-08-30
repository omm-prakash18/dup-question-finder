"""
src/features/feature_engineering.py
─────────────────────────────────────────────────────────────────────────────
STEP 3: Feature Engineering for Classical Baseline

Features extracted for each (question1, question2) pair:

GROUP A — Lexical / Fuzzy matching  (rapidfuzz)
  • fuzz_ratio            – simple character-level edit distance similarity
  • fuzz_partial_ratio    – best alignment of shorter string inside longer
  • fuzz_token_sort_ratio – order-independent token matching
  • fuzz_token_set_ratio  – handles repeated tokens well

GROUP B — TF-IDF cosine similarity
  • tfidf_cosine_q1_q2    – fit on training corpus; query/document similarity

GROUP C — Word/char count statistics
  • len_q1_chars, len_q2_chars, len_diff_chars
  • len_q1_words, len_q2_words, len_diff_words

GROUP D — Vocabulary overlap
  • common_word_ratio     – |Q1 ∩ Q2| / |Q1 ∪ Q2|  (Jaccard on words)
  • common_word_count

GROUP E — Word embedding similarity  (averaged word vectors via spaCy)
  • wv_cosine_similarity  – cosine between mean word vectors

WHY this set?
  These features are cheap to compute, interpretable, and between them cover
  surface form (A), distributional frequency (B), length bias (C), vocabulary
  overlap (D), and semantic proximity (E). XGBoost can then learn non-linear
  combinations of all five groups.
"""

import logging
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

PROCESSED_DIR = Path(CFG["paths"]["processed_data"])
MODEL_DIR = Path(CFG["paths"]["model_dir"])
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ── GROUP A: Fuzzy Matching Features ─────────────────────────────────────────
def fuzzy_features(q1: str, q2: str) -> dict:
    """
    Compute rapidfuzz similarity scores.
    All scores are in [0, 100]; we normalize to [0, 1] for consistency.
    """
    return {
        "fuzz_ratio":            fuzz.ratio(q1, q2) / 100,
        "fuzz_token_sort_ratio": fuzz.token_sort_ratio(q1, q2) / 100,
        "fuzz_token_set_ratio":  fuzz.token_set_ratio(q1, q2) / 100,
    }


def batch_fuzzy_features(df: pd.DataFrame, q1_col: str, q2_col: str) -> pd.DataFrame:
    """Vectorized wrapper; returns DataFrame of fuzzy features."""
    records = [
        fuzzy_features(q1, q2)
        for q1, q2 in zip(df[q1_col], df[q2_col])
    ]
    return pd.DataFrame(records, index=df.index)


# ── GROUP B: TF-IDF Cosine Similarity ────────────────────────────────────────
class TFIDFSimilarity:
    """
    Fits a TF-IDF vocabulary on the training corpus (union of Q1 and Q2),
    then at inference time computes the cosine similarity between the two
    TF-IDF vectors for each pair.

    WHY fit on training only? Prevents test-set vocabulary from leaking
    into the feature extractor — a classic form of data leakage.
    """

    def __init__(self, max_features=50_000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,     # log(tf) — dampens effect of very frequent tokens
            strip_accents="unicode",
            analyzer="word",
        )

    def fit(self, train_df: pd.DataFrame, q1_col: str, q2_col: str):
        """Fit on the combined vocabulary of Q1 and Q2 from the training set."""
        corpus = pd.concat([train_df[q1_col], train_df[q2_col]], ignore_index=True)
        self.vectorizer.fit(corpus)
        logger.info(
            f"TF-IDF fitted. Vocab size: {len(self.vectorizer.vocabulary_):,}"
        )
        return self

    def transform(self, df: pd.DataFrame, q1_col: str, q2_col: str) -> pd.Series:
        """Return per-pair cosine similarity as a Series."""
        mat_q1 = self.vectorizer.transform(df[q1_col])
        mat_q2 = self.vectorizer.transform(df[q2_col])
        # Row-wise dot product of L2-normalized TF-IDF vectors (elementwise multiply sum)
        scores = np.asarray(mat_q1.multiply(mat_q2).sum(axis=1)).ravel()
        return pd.Series(scores, index=df.index, name="tfidf_cosine")

    def save(self, path: Path):
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path: Path) -> "TFIDFSimilarity":
        obj = cls.__new__(cls)
        obj.vectorizer = joblib.load(path)
        return obj


# ── GROUP C: Length-based Features ───────────────────────────────────────────
def length_features(df: pd.DataFrame, q1_col: str, q2_col: str) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    feats["len_q1_chars"] = df[q1_col].str.len()
    feats["len_q2_chars"] = df[q2_col].str.len()
    feats["len_diff_chars"] = (feats["len_q1_chars"] - feats["len_q2_chars"]).abs()
    feats["len_q1_words"] = df[q1_col].str.split().str.len()
    feats["len_q2_words"] = df[q2_col].str.split().str.len()
    feats["len_diff_words"] = (feats["len_q1_words"] - feats["len_q2_words"]).abs()
    return feats


# ── GROUP D: Word Overlap / Jaccard ──────────────────────────────────────────
def overlap_features(df: pd.DataFrame, q1_col: str, q2_col: str) -> pd.DataFrame:
    ratios, counts = [], []
    for q1, q2 in zip(df[q1_col], df[q2_col]):
        s1 = set(str(q1).lower().split())
        s2 = set(str(q2).lower().split())
        inter = len(s1 & s2)
        union = len(s1 | s2)
        ratios.append(inter / union if union > 0 else 0.0)
        counts.append(inter)
    return pd.DataFrame({
        "common_word_ratio": ratios,
        "common_word_count": counts
    }, index=df.index)


# ── GROUP E: Word-Embedding Cosine Similarity (spaCy) ────────────────────────
def load_spacy_model():
    """
    Load spaCy's medium English model which includes GloVe word vectors.
    If not installed, return None gracefully.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_md")  # 20K vocab, 300-dim vectors
        return nlp
    except (ImportError, ModuleNotFoundError, OSError):
        logger.warning(
            "spaCy model not available. Word vector features will be skipped."
        )
        return None


def embedding_features(
    df: pd.DataFrame, q1_col: str, q2_col: str, nlp
) -> pd.DataFrame:
    """
    Mean pooled word vectors from spaCy → cosine similarity.
    Falls back to 0.0 if either question has no in-vocabulary tokens.

    WHY mean pooling? It's the simplest baseline for sentence-level semantics.
    It's intentionally weak — this is WHY we upgrade to SBERT in Stage 4.
    """
    feats = pd.DataFrame(index=df.index)
    if nlp is None:
        feats["wv_cosine_similarity"] = 0.0
        return feats

    sims = []
    for q1, q2 in zip(df[q1_col], df[q2_col]):
        doc1 = nlp(q1)
        doc2 = nlp(q2)
        # doc.vector is the mean of token vectors; has_vector checks for OOV
        if doc1.has_vector and doc2.has_vector:
            sim = doc1.similarity(doc2)
        else:
            sim = 0.0
        sims.append(float(sim))

    feats["wv_cosine_similarity"] = sims
    return feats


# ── Master Feature Builder ───────────────────────────────────────────────────
def build_features(
    df: pd.DataFrame,
    q1_col: str = "q1_classical",
    q2_col: str = "q2_classical",
    tfidf_sim: TFIDFSimilarity = None,
    nlp=None,
) -> pd.DataFrame:
    """
    Compose all feature groups into a single feature matrix.
    Returns a DataFrame whose columns are the feature names.
    """
    feature_parts = [
        batch_fuzzy_features(df, q1_col, q2_col),
        length_features(df, q1_col, q2_col),
        overlap_features(df, q1_col, q2_col),
    ]

    if tfidf_sim is not None:
        tfidf_col = tfidf_sim.transform(df, q1_col, q2_col)
        feature_parts.append(tfidf_col.to_frame())

    if nlp is not None:
        feature_parts.append(embedding_features(df, q1_col, q2_col, nlp))

    features = pd.concat(feature_parts, axis=1)
    return features


if __name__ == "__main__":
    """Quick smoke test."""
    logging.basicConfig(level=logging.INFO)
    train = pd.read_parquet(PROCESSED_DIR / "train_cleaned.parquet")
    sample = train.head(500)

    tfidf = TFIDFSimilarity(max_features=50_000)
    tfidf.fit(train, "q1_classical", "q2_classical")
    tfidf.save(MODEL_DIR / "tfidf_vectorizer.pkl")

    nlp = load_spacy_model()
    feats = build_features(sample, tfidf_sim=tfidf, nlp=nlp)
    print(f"Feature matrix shape: {feats.shape}")
    print(feats.head())
