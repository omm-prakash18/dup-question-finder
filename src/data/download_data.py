"""
src/data/download_data.py
─────────────────────────────────────────────────────────────────────────────
STEP 1: Data Acquisition
Downloads the HuggingFace `quora` dataset, inspects its schema, deduplicates
the question corpus, and saves processed splits to disk.

WHY quora (HF) instead of Kaggle?
  - Available without a Kaggle account via `datasets` library.
  - Same source data (~400K pairs, binary is_duplicate label).
  - Fully reproducible with a single `load_dataset` call.
"""

import os
import json
import sqlite3
import logging
from pathlib import Path

import yaml
import pandas as pd
from datasets import load_dataset

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Load central config
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

RAW_DIR = Path(CFG["paths"]["raw_data"])
PROCESSED_DIR = Path(CFG["paths"]["processed_data"])
FAISS_DIR = Path(CFG["paths"]["faiss_dir"])
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load Dataset ───────────────────────────────────────────────────────────
def load_quora_dataset():
    """
    Downloads the Quora Question Pairs dataset via HuggingFace `datasets`.
    Primary source: `glue/qqp` (official HuggingFace CDN mirror).
    Fallback: `quora`
    """
    logger.info("Downloading HuggingFace dataset ('glue', 'qqp')...")
    try:
        ds = load_dataset("glue", "qqp", split="train")
    except Exception as e:
        logger.warning(f"Could not load GLUE QQP ({e}), trying 'quora'...")
        ds = load_dataset("quora", split="train", trust_remote_code=True)

    logger.info(f"Loaded {len(ds):,} question pairs.")
    logger.info(f"Features: {ds.features}")

    # Show a few examples
    logger.info("\n── Sample rows ──────────────────────────────────────")
    for i in range(min(3, len(ds))):
        ex = ds[i]
        q1 = ex.get("question1") or ex.get("questions", {}).get("text", ["", ""])[0]
        q2 = ex.get("question2") or ex.get("questions", {}).get("text", ["", ""])[1]
        label = ex.get("label") if "label" in ex else ex.get("is_duplicate")
        logger.info(f"Row {i}: label={label} | Q1: '{q1[:50]}' | Q2: '{q2[:50]}'")
    return ds


# ── 2. Flatten & Deduplicate Question Corpus ─────────────────────────────────
def build_question_corpus(ds) -> pd.DataFrame:
    """
    Extract unique questions from the dataset pairs for FAISS indexing.
    Returns a DataFrame: [question_id, question_text]
    """
    logger.info("Building unique question corpus from pairs...")
    id_to_text = {}
    seen_texts = set()

    for idx, example in enumerate(ds):
        if "question1" in example and "question2" in example:
            q1, q2 = str(example["question1"]).strip(), str(example["question2"]).strip()
            qid1 = example.get("idx", idx * 2) * 2
            qid2 = example.get("idx", idx * 2) * 2 + 1
            if q1 and q1 not in seen_texts:
                seen_texts.add(q1)
                id_to_text[qid1] = q1
            if q2 and q2 not in seen_texts:
                seen_texts.add(q2)
                id_to_text[qid2] = q2
        elif "questions" in example:
            ids = example["questions"]["id"]
            texts = example["questions"]["text"]
            for qid, qtxt in zip(ids, texts):
                txt = str(qtxt).strip()
                if qid not in id_to_text and txt and txt not in seen_texts:
                    seen_texts.add(txt)
                    id_to_text[qid] = txt

    corpus_df = pd.DataFrame(
        list(id_to_text.items()), columns=["question_id", "question_text"]
    )
    logger.info(f"Unique questions in corpus: {len(corpus_df):,}")
    return corpus_df


# ── 3. Build Pairs DataFrame (for classification training) ────────────────────
def build_pairs_dataframe(ds) -> pd.DataFrame:
    """
    Flatten question pairs into a standard DataFrame for ML training.
    Returns columns: [qid1, qid2, question1, question2, is_duplicate]
    """
    logger.info("Building pairs DataFrame...")
    rows = []

    for idx, example in enumerate(ds):
        if "question1" in example and "question2" in example:
            q1 = str(example["question1"]).strip()
            q2 = str(example["question2"]).strip()
            label = int(example.get("label", 0))
            if q1 and q2:
                rows.append({
                    "qid1": example.get("idx", idx * 2) * 2,
                    "qid2": example.get("idx", idx * 2) * 2 + 1,
                    "question1": q1,
                    "question2": q2,
                    "is_duplicate": label,
                })
        elif "questions" in example:
            q1 = str(example["questions"]["text"][0]).strip()
            q2 = str(example["questions"]["text"][1]).strip()
            label = int(example["is_duplicate"])
            if q1 and q2:
                rows.append({
                    "qid1": example["questions"]["id"][0],
                    "qid2": example["questions"]["id"][1],
                    "question1": q1,
                    "question2": q2,
                    "is_duplicate": label,
                })

    df = pd.DataFrame(rows)
    df = df[(df["question1"].str.len() > 0) & (df["question2"].str.len() > 0)]
    logger.info(
        f"Total pairs: {len(df):,} | "
        f"Duplicates: {df['is_duplicate'].sum():,} "
        f"({df['is_duplicate'].mean()*100:.1f}%)"
    )
    return df


# ── 4. Train / Val / Test Split ──────────────────────────────────────────────
def split_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split to preserve duplicate/non-duplicate ratio.

    WHY stratified? The quora dataset is class-imbalanced (~37% duplicates).
    A random split could yield a training set with very different class balance.
    """
    from sklearn.model_selection import train_test_split

    seed = CFG["dataset"]["seed"]
    test_sz = CFG["dataset"]["test_size"]
    val_sz = CFG["dataset"]["val_size"]

    # First split off test set
    train_val, test = train_test_split(
        df, test_size=test_sz, stratify=df["is_duplicate"], random_state=seed
    )
    # Then split val from train
    val_relative = val_sz / (1 - test_sz)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val["is_duplicate"],
        random_state=seed,
    )

    logger.info(
        f"Split sizes — Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}"
    )
    return train, val, test


# ── 5. Persist to Disk ───────────────────────────────────────────────────────
def save_splits(
    corpus: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
):
    """Save all DataFrames as Parquet (fast, columnar, preserves dtypes)."""
    corpus.to_parquet(PROCESSED_DIR / "corpus.parquet", index=False)
    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    logger.info(f"Saved all splits to {PROCESSED_DIR}")


def save_corpus_to_sqlite(corpus: pd.DataFrame):
    """
    Also persist question corpus to SQLite so the API can do O(1) lookups
    by question_id without loading the full Parquet file into memory.
    """
    db_path = FAISS_DIR / CFG["faiss"]["metadata_db"]
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS questions (
            question_id   INTEGER PRIMARY KEY,
            question_text TEXT NOT NULL,
            faiss_idx     INTEGER,          -- assigned after FAISS index build
            source        TEXT DEFAULT 'quora_hf'
        )
        """
    )
    # Use pandas to_sql for bulk insert (much faster than row-by-row)
    corpus.to_sql(
        "questions", conn, if_exists="replace", index=False,
        dtype={"question_id": "INTEGER", "question_text": "TEXT"},
    )
    conn.commit()
    conn.close()
    logger.info(f"SQLite metadata store saved to {db_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = load_quora_dataset()
    df_raw = ds.to_pandas()
    corpus_df = build_question_corpus(df_raw)
    pairs_df = build_pairs_dataframe(df_raw)
    train, val, test = split_dataframe(pairs_df)
    save_splits(corpus_df, train, val, test)
    save_corpus_to_sqlite(corpus_df)

    # Print schema summary for the developer
    print("\n" + "=" * 60)
    print("DATA ACQUISITION COMPLETE")
    print("=" * 60)
    print(f"  Unique questions (corpus): {len(corpus_df):,}")
    print(f"  Train pairs:               {len(train):,}")
    print(f"  Val pairs:                 {len(val):,}")
    print(f"  Test pairs:                {len(test):,}")
    print(f"  Class balance (train):     {train['is_duplicate'].mean()*100:.1f}% duplicates")
    print(f"\nFiles saved in: {PROCESSED_DIR}")
    print(f"SQLite DB:      {FAISS_DIR / CFG['faiss']['metadata_db']}")
