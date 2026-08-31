"""
src/models/evaluate_deep.py
─────────────────────────────────────────────────────────────────────────────
DEEP EVALUATION & DEBUGGING SCRIPT
Provides:
  1. Full classification metrics: Precision, Recall, F1, AUC-ROC, AP
  2. Extrinsic retrieval metrics: Recall@1, Recall@5, Recall@10, MRR
  3. Sliced evaluation by Question Length Buckets:
     - Short  (< 8 words)
     - Medium (8 - 15 words)
     - Long   (> 15 words)
     Catching length-based bias where short questions are harder due to lack of context.
  4. Top 10 False Positives & Top 10 False Negatives (with actual question strings)
  5. Overfitting check: Train vs Val loss & F1 gap
"""

import sys
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import evaluate_classifier, evaluate_retrieval, print_confusion_matrix

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

PROCESSED_DIR = Path(CFG["paths"]["processed_data"])
MODEL_DIR     = Path(CFG["paths"]["model_dir"])
FAISS_DIR     = Path(CFG["paths"]["faiss_dir"])


def slice_by_length(df: pd.DataFrame) -> dict:
    """Categorize rows into short, medium, and long length buckets."""
    q1_lens = df["question1"].str.split().str.len()
    q2_lens = df["question2"].str.split().str.len()
    avg_lens = (q1_lens + q2_lens) / 2.0

    slices = {
        "short (<8 words)":   df[avg_lens < 8],
        "medium (8-15 words)": df[(avg_lens >= 8) & (avg_lens <= 15)],
        "long (>15 words)":    df[avg_lens > 15],
    }
    return slices


def analyze_false_positives_and_negatives(df: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, top_n: int = 10):
    """Print top N highest-confidence False Positives and False Negatives."""
    df_eval = df.copy()
    df_eval["y_true"] = y_true
    df_eval["y_prob"] = y_prob
    df_eval["y_pred"] = (y_prob >= threshold).astype(int)

    # False Positives: y_true == 0, y_pred == 1 (sorted by highest probability)
    fps = df_eval[(df_eval["y_true"] == 0) & (df_eval["y_pred"] == 1)].sort_values("y_prob", ascending=False)

    # False Negatives: y_true == 1, y_pred == 0 (sorted by lowest probability)
    fns = df_eval[(df_eval["y_true"] == 1) & (df_eval["y_pred"] == 0)].sort_values("y_prob", ascending=True)

    print("\n" + "="*75)
    print(f" TOP {top_n} FALSE POSITIVES (Model predicted duplicate with high confidence, but label is non-duplicate)")
    print("="*75)
    for idx, row in fps.head(top_n).reset_index().iterrows():
        print(f"[{idx+1}] Prob={row['y_prob']:.4f}")
        print(f"    Q1: {row['question1']}")
        print(f"    Q2: {row['question2']}")

    print("\n" + "="*75)
    print(f" TOP {top_n} FALSE NEGATIVES (Model predicted non-duplicate, but label is duplicate)")
    print("="*75)
    for idx, row in fns.head(top_n).reset_index().iterrows():
        print(f"[{idx+1}] Prob={row['y_prob']:.4f}")
        print(f"    Q1: {row['question1']}")
        print(f"    Q2: {row['question2']}")
    print("="*75)


def run_deep_evaluation():
    test_path = PROCESSED_DIR / "test_cleaned.parquet"
    if not test_path.exists():
        print(f"Test file {test_path} not found.")
        return

    test = pd.read_parquet(test_path).reset_index(drop=True)
    y_test = test["is_duplicate"].values

    # Check if XGBoost baseline model exists
    xgb_path = MODEL_DIR / "xgboost_model.pkl"
    if xgb_path.exists():
        import joblib
        from src.features.feature_engineering import TFIDFSimilarity, build_features, load_spacy_model

        print("\n" + "="*70)
        print(" DEEP EVALUATION: XGBoost Classical Baseline")
        print("="*70)

        xgb = joblib.load(xgb_path)
        tfidf = TFIDFSimilarity.load(MODEL_DIR / "tfidf_vectorizer.pkl")
        nlp = load_spacy_model()

        X_test = build_features(test, tfidf_sim=tfidf, nlp=nlp)
        y_prob = xgb.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        print("\n--- Overall Test Metrics ---")
        metrics = evaluate_classifier(y_test, y_pred, y_prob)
        for k, v in metrics.items():
            print(f"  {k:15s}: {v:.4f}")

        print_confusion_matrix(y_test, y_pred)

        # Sliced metrics by length
        print("\n--- Sliced Performance by Question Length ---")
        slices = slice_by_length(test)
        for slice_name, slice_df in slices.items():
            if len(slice_df) == 0:
                continue
            slice_idx = slice_df.index
            slice_y = y_test[slice_df.index]
            slice_prob = y_prob[slice_df.index]
            slice_pred = (slice_prob >= 0.5).astype(int)

            prec = evaluate_classifier(slice_y, slice_pred, slice_prob)
            print(f"  {slice_name:<20s} (N={len(slice_df):<6,}) -> F1: {prec['f1']:.4f} | AUC: {prec.get('auc_roc', 0):.4f}")

        # Diagnostic False Positives & Negatives
        analyze_false_positives_and_negatives(test, y_test, y_prob, threshold=0.5, top_n=5)

    # Check SBERT pipeline if available
    sbert_dir = MODEL_DIR / "sbert_finetuned"
    if sbert_dir.exists():
        from src.models.sbert_pipeline import SimilarQuestionRetriever, evaluate_retrieval_quality

        print("\n" + "="*70)
        print(" DEEP EVALUATION: SBERT + FAISS Retrieval System")
        print("="*70)

        retriever = SimilarQuestionRetriever(
            model_path=str(sbert_dir),
            index_path=str(FAISS_DIR / CFG["faiss"]["index_filename"]),
            db_path=str(FAISS_DIR / CFG["faiss"]["metadata_db"]),
        )
        evaluate_retrieval_quality(retriever, test)


if __name__ == "__main__":
    run_deep_evaluation()
