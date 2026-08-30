"""
src/data/preprocess.py
─────────────────────────────────────────────────────────────────────────────
STEP 2: Text Preprocessing Pipeline

TWO modes:
  - classical: full cleaning (lowercasing, contractions, punctuation removal,
    stopword removal) — needed for TF-IDF which is bag-of-words sensitive
  - transformer: minimal cleaning (HTML/URL removal only) — BERT/SBERT have
    their own subword tokenizer that handles casing, punctuation, etc.

WHY skip stemming/stopwords for transformers?
  SBERT's tokenizer (WordPiece) uses the raw surface form of words.
  Stemming "running" → "run" throws away tense information that the model
  uses to build contextual embeddings. Transformers are also case-sensitive
  by design (BERT-cased variants) or robustly handle mixed case via their
  own normalization.
"""

import re
import html
import logging
from pathlib import Path

import yaml
import pandas as pd
import contractions

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

PROCESSED_DIR = Path(CFG["paths"]["processed_data"])

# Compile regexes once at module load (performance)
_URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_PUNCT_RE = re.compile(r"[^\w\s]")   # keep words and spaces
_MULTI_SPACE_RE = re.compile(r"\s+")


def remove_html(text: str) -> str:
    """
    Unescape HTML entities (&amp; → &) then strip tags and normalize whitespace.
    Quora questions sometimes come with &amp;, &quot;, etc.
    """
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def remove_urls(text: str) -> str:
    """Replace URLs with a placeholder token so the model sees *something*
    at that position rather than a gap. Token '<URL>' is rare enough to be
    informative without polluting the vocabulary."""
    return _URL_RE.sub("<URL>", text)


def expand_contractions_text(text: str) -> str:
    """
    Expand English contractions: don't → do not, it's → it is.
    Only invokes heavy contractions engine if an apostrophe is detected.
    """
    if "'" in text or "’" in text or "`" in text:
        return contractions.fix(text)
    return text


def clean_classical(text: str) -> str:
    """
    Full pipeline for classical ML (TF-IDF + XGBoost).
    Produces a clean, lowercase, punctuation-free string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = remove_html(text)
    text = remove_urls(text)
    text = expand_contractions_text(text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def clean_transformer(text: str) -> str:
    """
    Minimal pipeline for transformer-based models (SBERT).
    Preserves: casing, punctuation, word morphology.
    Removes: only HTML tags and raw URLs (noise, not language).
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = remove_html(text)
    text = remove_urls(text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def apply_cleaning(df: pd.DataFrame, mode: str = "transformer") -> pd.DataFrame:
    """
    Apply the appropriate cleaning function to question1 and question2 columns.

    Args:
        df:   DataFrame with columns [question1, question2, ...]
        mode: 'classical' or 'transformer'

    Returns:
        DataFrame with cleaned question columns (new columns to avoid overwriting)
    """
    clean_fn = clean_classical if mode == "classical" else clean_transformer

    logger.info(f"Applying '{mode}' cleaning pipeline...")
    df = df.copy()
    df[f"q1_{mode}"] = df["question1"].apply(clean_fn)
    df[f"q2_{mode}"] = df["question2"].apply(clean_fn)

    # Drop rows where cleaning produced an empty string
    before = len(df)
    df = df[
        (df[f"q1_{mode}"].str.len() > 0) &
        (df[f"q2_{mode}"].str.len() > 0)
    ]
    logger.info(f"Dropped {before - len(df)} empty rows after cleaning.")
    return df


def preprocess_and_save():
    """Load raw splits, apply both cleaning modes, save."""
    for split in ["train", "val", "test"]:
        path = PROCESSED_DIR / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run src/data/download_data.py first."
            )
        df = pd.read_parquet(path)

        df = apply_cleaning(df, mode="classical")
        df = apply_cleaning(df, mode="transformer")

        out_path = PROCESSED_DIR / f"{split}_cleaned.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Saved cleaned {split} split → {out_path}")

    # Also clean the corpus for FAISS embedding
    corpus = pd.read_parquet(PROCESSED_DIR / "corpus.parquet")
    corpus["question_clean"] = corpus["question_text"].apply(clean_transformer)
    corpus.to_parquet(PROCESSED_DIR / "corpus_cleaned.parquet", index=False)
    logger.info("Saved cleaned corpus → corpus_cleaned.parquet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    preprocess_and_save()
    print("Preprocessing complete.")
