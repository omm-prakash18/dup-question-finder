"""
src/models/baseline.py
─────────────────────────────────────────────────────────────────────────────
STEP 4a: Classical ML Baseline
  TF-IDF features + hand-crafted features → XGBoost binary classifier

WHY XGBoost over Logistic Regression as the primary baseline?
  - Handles non-linear feature interactions (e.g., high fuzz_ratio AND
    high length_diff might together signal a near-duplicate, not alone).
  - Robust to outliers and missing values.
  - Built-in feature importance for interpretability.
  - We also train Logistic Regression for comparison.
"""

import logging
import json
import joblib
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Internal imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_engineering import (
    TFIDFSimilarity, build_features, load_spacy_model
)
from src.evaluation.metrics import evaluate_classifier, print_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

PROCESSED_DIR = Path(CFG["paths"]["processed_data"])
MODEL_DIR = Path(CFG["paths"]["model_dir"])
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_splits():
    logger.info("Loading cleaned train/val/test splits...")
    train = pd.read_parquet(PROCESSED_DIR / "train_cleaned.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / "val_cleaned.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / "test_cleaned.parquet")
    return train, val, test


def build_all_features(train, val, test, nlp=None):
    """
    Fit the TF-IDF vectorizer on training data only, then transform all splits.
    This is the key discipline: no look-ahead at val/test vocabulary.
    """
    tfidf = TFIDFSimilarity(
        max_features=CFG["baseline"]["tfidf_max_features"],
        ngram_range=tuple(CFG["baseline"]["tfidf_ngram_range"]),
    )
    tfidf.fit(train, "q1_classical", "q2_classical")
    tfidf.save(MODEL_DIR / "tfidf_vectorizer.pkl")

    logger.info("Computing features for train, val, test...")
    X_train = build_features(train, tfidf_sim=tfidf, nlp=nlp)
    X_val   = build_features(val,   tfidf_sim=tfidf, nlp=nlp)
    X_test  = build_features(test,  tfidf_sim=tfidf, nlp=nlp)

    y_train = train["is_duplicate"].values
    y_val   = val["is_duplicate"].values
    y_test  = test["is_duplicate"].values

    logger.info(f"Feature matrix shape (train): {X_train.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Logistic Regression with L2 regularization.
    Features need to be standardized for LR (gradient-based solver).
    """
    logger.info("Training Logistic Regression baseline...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    lr = LogisticRegression(
        C=1.0,            # inverse regularization strength
        max_iter=1000,
        class_weight="balanced",  # compensate for class imbalance
        solver="lbfgs",
        n_jobs=-1,
    )
    lr.fit(X_train_sc, y_train)

    y_prob = lr.predict_proba(X_val_sc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = evaluate_classifier(y_val, y_pred, y_prob)
    logger.info(f"[LR] Val metrics: {metrics}")
    print_confusion_matrix(y_val, y_pred)

    joblib.dump({"model": lr, "scaler": scaler}, MODEL_DIR / "logistic_regression.pkl")
    return lr, scaler, metrics


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    XGBoost with early stopping on validation AUC.

    Key hyperparameters (and why):
    - max_depth=6: Deep enough for interaction detection, not so deep it overfits.
    - learning_rate=0.05: Small LR + many trees = better generalization than
      high LR + few trees (gradient boosting bias-variance trade-off).
    - scale_pos_weight: ratio of negatives to positives, compensates imbalance.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos  # ~1.7 for quora (~37% duplicate)
    logger.info(
        f"Training XGBoost: pos={pos:,}, neg={neg:,}, "
        f"scale_pos_weight={scale_pos_weight:.2f}"
    )

    xgb = XGBClassifier(
        n_estimators=CFG["baseline"]["xgb_n_estimators"],
        max_depth=CFG["baseline"]["xgb_max_depth"],
        learning_rate=CFG["baseline"]["xgb_learning_rate"],
        subsample=CFG["baseline"]["xgb_subsample"],
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="auc",          # watch AUC on validation
        early_stopping_rounds=30,   # stop if no improvement in 30 rounds
        tree_method="hist",         # GPU-accelerated histogram method
        device="cuda",              # use your NVIDIA GPU
        random_state=CFG["dataset"]["seed"],
        n_jobs=-1,
    )

    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,                 # print every 50 rounds
    )

    logger.info(f"Best iteration: {xgb.best_iteration}")

    y_prob = xgb.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = evaluate_classifier(y_val, y_pred, y_prob)
    logger.info(f"[XGB] Val metrics: {metrics}")
    print_confusion_matrix(y_val, y_pred)

    # Feature importance — helps understand which signals matter most
    feat_names = X_train.columns.tolist() if hasattr(X_train, "columns") else None
    if feat_names:
        importances = dict(zip(feat_names, xgb.feature_importances_))
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        logger.info("\nTop 10 feature importances:")
        for name, imp in sorted_imp[:10]:
            logger.info(f"  {name:35s}: {imp:.4f}")

    joblib.dump(xgb, MODEL_DIR / "xgboost_model.pkl")
    return xgb, metrics


def evaluate_on_test(model, X_test, y_test, scaler=None, model_name="Model"):
    """Final evaluation on held-out test set — only called once."""
    if scaler is not None:
        X_test = scaler.transform(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = evaluate_classifier(y_test, y_pred, y_prob)
    print(f"\n{'='*50}")
    print(f"FINAL TEST RESULTS — {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    return metrics


if __name__ == "__main__":
    train, val, test = load_splits()

    # Load spaCy (optional; skip if en_core_web_md not installed)
    nlp = load_spacy_model()

    X_train, X_val, X_test, y_train, y_val, y_test = build_all_features(
        train, val, test, nlp=nlp
    )

    # --- Classical baseline 1: Logistic Regression ---
    lr_model, scaler, lr_val_metrics = train_logistic_regression(
        X_train, y_train, X_val, y_val
    )

    # --- Classical baseline 2: XGBoost ---
    xgb_model, xgb_val_metrics = train_xgboost(X_train, y_train, X_val, y_val)

    # --- Final test evaluation ---
    lr_test_metrics  = evaluate_on_test(lr_model,  X_test, y_test, scaler, "Logistic Regression")
    xgb_test_metrics = evaluate_on_test(xgb_model, X_test, y_test, model_name="XGBoost")

    # Save results summary
    results = {
        "logistic_regression": {"val": lr_val_metrics, "test": lr_test_metrics},
        "xgboost": {"val": xgb_val_metrics, "test": xgb_test_metrics},
    }
    with open(MODEL_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {MODEL_DIR / 'baseline_results.json'}")
