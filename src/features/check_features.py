"""
src/features/check_features.py
─────────────────────────────────────────────────────────────────────────────
FEATURE CORRELATION & SIGNAL DIAGNOSTIC SCRIPT
Computes point-biserial correlation between engineered features and target `is_duplicate`.
Flags any feature with near-zero correlation (|r| < 0.05).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

PROCESSED_DIR = Path(CFG["paths"]["processed_data"])
MODEL_DIR     = Path(CFG["paths"]["model_dir"])

# Add root to import path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_engineering import TFIDFSimilarity, build_features, load_spacy_model


def analyze_feature_correlations():
    train_path = PROCESSED_DIR / "train_cleaned.parquet"
    if not train_path.exists():
        print("Cleaned dataset not found. Run preprocess.py first.")
        return

    print("Loading sample of training dataset...")
    df = pd.read_parquet(train_path).sample(min(10000, len(pd.read_parquet(train_path))), random_state=42)

    # Load TF-IDF vectorizer if exists, else fit temporary one
    tfidf_path = MODEL_DIR / "tfidf_vectorizer.pkl"
    if tfidf_path.exists():
        tfidf = TFIDFSimilarity.load(tfidf_path)
    else:
        tfidf = TFIDFSimilarity(max_features=10000).fit(df, "q1_classical", "q2_classical")

    nlp = load_spacy_model()

    print("Generating feature matrix...")
    X = build_features(df, q1_col="q1_classical", q2_col="q2_classical", tfidf_sim=tfidf, nlp=nlp)
    y = df["is_duplicate"].values

    print("\n" + "="*70)
    print(" POINT-BISERIAL CORRELATION WITH TARGET (is_duplicate)")
    print("="*70)
    print(f"{'Feature Name':<30} | {'Correlation r':<15} | {'p-value':<12} | Status")
    print("-" * 70)

    for col in X.columns:
        # Fill any residual NaNs with 0
        vals = np.nan_to_num(X[col].values)
        r, pval = pointbiserialr(vals, y)
        status = "⚠️ LOW SIGNAL" if abs(r) < 0.05 else "✅ OK"
        print(f"{col:<30} | {r:>14.4f} | {pval:>11.2e} | {status}")

    print("="*70)


if __name__ == "__main__":
    analyze_feature_correlations()
