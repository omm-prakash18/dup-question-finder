"""
src/evaluation/metrics.py
─────────────────────────────────────────────────────────────────────────────
STEP 4: Evaluation Utilities

Provides two types of evaluation:

INTRINSIC (classification):
  • Precision, Recall, F1 (at threshold 0.5)
  • AUC-ROC (threshold-free)
  • Confusion matrix

EXTRINSIC (retrieval):
  • Recall@K  — "did the true duplicate appear in the top-K results?"
  • MRR       — Mean Reciprocal Rank; rewards finding the duplicate earlier

WHY both?
  Intrinsic tells you how good the model is as a binary classifier.
  Extrinsic tells you how good the *system* is for the user-facing task
  of "show me the K most similar questions." AUC-ROC doesn't capture
  whether a true duplicate is ranked 1st vs 5th; MRR and Recall@K do.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
)

logger = logging.getLogger(__name__)


# ── INTRINSIC METRICS ─────────────────────────────────────────────────────────
def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true:    Ground truth binary labels
        y_pred:    Predicted binary labels (at given threshold)
        y_prob:    Predicted probabilities for positive class (for AUC)
        threshold: Decision threshold for converting probabilities to labels

    Returns:
        dict with precision, recall, f1, auc_roc, avg_precision
    """
    if y_prob is not None and y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )

    metrics = {
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f1":        round(float(f1), 4),
    }

    if y_prob is not None:
        metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        # Average Precision = area under PR curve (useful for imbalanced data)
        metrics["avg_precision"] = round(
            float(average_precision_score(y_true, y_prob)), 4
        )

    logger.info(f"\n{classification_report(y_true, y_pred)}")
    return metrics


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:>6,}  FP={fp:>6,}")
    print(f"  FN={fn:>6,}  TP={tp:>6,}")


# ── EXTRINSIC METRICS (Retrieval) ─────────────────────────────────────────────
def recall_at_k(retrieved_ids_list: list[list], relevant_ids_list: list[list], k: int) -> float:
    """
    Recall@K: For each query, did at least one relevant item appear in top-K?

    Args:
        retrieved_ids_list: List of lists — for each query, the IDs retrieved (ordered)
        relevant_ids_list:  List of lists — for each query, the ground-truth relevant IDs
        k:                  Cutoff rank

    Returns:
        Float in [0, 1]
    """
    hits = 0
    for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
        top_k = set(retrieved[:k])
        if top_k & set(relevant):  # intersection non-empty
            hits += 1
    return hits / len(retrieved_ids_list) if retrieved_ids_list else 0.0


def mean_reciprocal_rank(retrieved_ids_list: list[list], relevant_ids_list: list[list]) -> float:
    """
    MRR: 1/rank of the first relevant item in the retrieved list.

    WHY MRR? If the correct answer is at rank 1 MRR=1.0; at rank 5 MRR=0.2.
    It's the standard metric for question-answering / retrieval systems.
    """
    rr_sum = 0.0
    for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
        relevant_set = set(relevant)
        for rank, item_id in enumerate(retrieved, start=1):
            if item_id in relevant_set:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(retrieved_ids_list) if retrieved_ids_list else 0.0


def evaluate_retrieval(
    retrieved_ids_list: list[list],
    relevant_ids_list: list[list],
    k_values: list[int] = [1, 5, 10],
) -> dict:
    """
    Compute all retrieval metrics for multiple K values.

    Args:
        retrieved_ids_list: Per-query retrieved IDs (ranked)
        relevant_ids_list:  Per-query ground-truth relevant IDs
        k_values:           List of K cutoffs to evaluate

    Returns:
        dict with recall@K for each K, plus MRR
    """
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = round(
            recall_at_k(retrieved_ids_list, relevant_ids_list, k), 4
        )
    metrics["mrr"] = round(
        mean_reciprocal_rank(retrieved_ids_list, relevant_ids_list), 4
    )
    return metrics


def print_retrieval_report(metrics: dict):
    print("\n=== Retrieval Metrics ===")
    for metric, value in metrics.items():
        print(f"  {metric:12s}: {value:.4f}")
    print("=" * 50)


# ── QUICK SANITY CHECKS ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Toy example: 4 queries
    retrieved = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    relevant  = [[2],       [4],       [99],       [11]]

    metrics = evaluate_retrieval(retrieved, relevant, k_values=[1, 3])
    print("Retrieval metrics:", metrics)
    # Recall@1: 1/4=0.25, Recall@3: 3/4=0.75, MRR: (1/2+1+0+1/2)/4=0.5
