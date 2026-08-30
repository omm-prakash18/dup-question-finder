"""
notebooks/03_sbert_experiments.py
─────────────────────────────────────────────────────────────────────────────
Experiment notebook for SBERT + FAISS — interactive exploration after
the main pipeline has been run (sbert_pipeline.py).

Covers:
  A. Visualizing embedding space (UMAP/t-SNE)
  B. Threshold sweep for binary classification from cosine similarity
  C. Speed benchmark: embedding latency vs batch size
  D. Failure mode analysis — top false positives and false negatives
  E. Cross-encoder re-ranking demo (improvement over bi-encoder alone)
"""

# %% Imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import time

sns.set_theme(style="darkgrid")
PROCESSED = Path("../data/processed")
FAISS_DIR = Path("../data/faiss")
MODEL_DIR = Path("../data/models")

# %% A. Load model and sample embeddings
model = SentenceTransformer(str(MODEL_DIR / "sbert_finetuned"), device="cuda")
test  = pd.read_parquet(PROCESSED / "test_cleaned.parquet")
sample = test.sample(2000, random_state=42)

print("Embedding 2000 question pairs...")
t0 = time.time()
embs_q1 = model.encode(sample["q1_transformer"].tolist(), batch_size=128,
                        normalize_embeddings=True, show_progress_bar=True)
embs_q2 = model.encode(sample["q2_transformer"].tolist(), batch_size=128,
                        normalize_embeddings=True, show_progress_bar=True)
print(f"Elapsed: {time.time()-t0:.1f}s")

# Cosine similarities (dot product on L2-normalized = cosine)
cos_sims = (embs_q1 * embs_q2).sum(axis=1)
sample = sample.copy()
sample["cosine_sim"] = cos_sims

# %% B. Threshold sweep — find optimal classification threshold
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curve
fpr, tpr, roc_thresholds = roc_curve(sample["is_duplicate"], sample["cosine_sim"])
roc_auc = auc(fpr, tpr)
axes[0].plot(fpr, tpr, color="#FF6B6B", lw=2, label=f"AUC = {roc_auc:.3f}")
axes[0].plot([0,1],[0,1], "k--", alpha=0.5)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — Fine-tuned SBERT")
axes[0].legend()

# Precision-Recall curve
prec, rec, pr_thresholds = precision_recall_curve(sample["is_duplicate"], sample["cosine_sim"])
f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
best_idx = f1_scores.argmax()
best_thresh = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.5
axes[1].plot(rec, prec, color="#5B8FF9", lw=2)
axes[1].axvline(rec[best_idx], color="#FF6B6B", linestyle="--",
                label=f"Best threshold = {best_thresh:.3f}\nF1 = {f1_scores[best_idx]:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()

plt.suptitle("SBERT Binary Classification Performance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../logs/sbert_roc_pr.png", dpi=150)
plt.show()
print(f"Optimal threshold: {best_thresh:.3f} → F1={f1_scores[best_idx]:.3f}")

# %% C. Embedding latency benchmark
print("\n── Embedding Speed Benchmark ──────────────────────────────────")
questions = ["How do I learn Python?" ] * 1000  # 1000 identical questions

for batch_size in [1, 16, 64, 128, 256]:
    t0 = time.perf_counter()
    _ = model.encode(questions[:256], batch_size=batch_size,
                     normalize_embeddings=True, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    print(f"  batch_size={batch_size:3d} | {elapsed*1000:.1f}ms total | "
          f"{elapsed*1000/256:.2f}ms/query")

# %% D. Failure mode analysis
sample["predicted_dup"] = (sample["cosine_sim"] >= best_thresh).astype(int)
sample["correct"] = sample["predicted_dup"] == sample["is_duplicate"]

# False Positives: predicted duplicate but not
fp = sample[(sample["predicted_dup"]==1) & (sample["is_duplicate"]==0)].nlargest(5, "cosine_sim")
print("\n── Top False Positives (model thinks duplicate, but isn't) ───")
for _, r in fp.iterrows():
    print(f"  Score={r['cosine_sim']:.3f}")
    print(f"  Q1: {r['question1'][:80]}")
    print(f"  Q2: {r['question2'][:80]}\n")

# False Negatives: predicted non-duplicate but actually is
fn = sample[(sample["predicted_dup"]==0) & (sample["is_duplicate"]==1)].nsmallest(5, "cosine_sim")
print("── Top False Negatives (model misses actual duplicates) ──────")
for _, r in fn.iterrows():
    print(f"  Score={r['cosine_sim']:.3f}")
    print(f"  Q1: {r['question1'][:80]}")
    print(f"  Q2: {r['question2'][:80]}\n")

# %% E. Cross-encoder re-ranking demo
# -------------------------------------------------------------------
# A cross-encoder reads BOTH questions simultaneously (joint attention),
# unlike the bi-encoder which reads them independently.
# This gives higher precision at the cost of O(K) inference per query
# (only feasible for re-ranking, not for the full corpus search).
# -------------------------------------------------------------------
print("\n── Cross-Encoder Re-ranking Demo ──────────────────────────────")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

demo_query = "What is the fastest way to learn machine learning?"
# Simulate bi-encoder retrieving top-10 candidates
candidates = [
    "How can I learn machine learning quickly?",            # true dup
    "What is the best resource for learning deep learning?", # related
    "How do I become a data scientist?",                    # somewhat related
    "What is machine learning?",                            # related but not dup
    "How long does it take to learn ML?",                  # related
]

# Bi-encoder scores (quick)
bi_emb_q  = model.encode([demo_query], normalize_embeddings=True)
bi_emb_cs = model.encode(candidates,  normalize_embeddings=True)
bi_scores  = (bi_emb_q @ bi_emb_cs.T).flatten()

# Cross-encoder scores (slow but more accurate)
ce_input  = [[demo_query, c] for c in candidates]
ce_scores = cross_encoder.predict(ce_input)

print(f"Query: '{demo_query}'\n")
print(f"{'Candidate':<55} {'Bi-enc':>7} {'Cross-enc':>9}")
print("-" * 75)
for cand, bi, ce in zip(candidates, bi_scores, ce_scores):
    print(f"  {cand[:52]:<52} {bi:>7.3f} {ce:>9.3f}")

# Cross-encoder re-ranking often changes the order, especially for
# semantically tricky pairs where surface-level features mislead the bi-encoder.
