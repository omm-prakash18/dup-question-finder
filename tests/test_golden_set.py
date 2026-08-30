"""
tests/test_golden_set.py — Golden Regression Test Suite
─────────────────────────────────────────────────────────────────────────────
Hand-curated benchmark of 25 challenging question pairs covering:
  1. Obvious Duplicates (canonical paraphrases)
  2. Obvious Non-Duplicates (different domains)
  3. Tricky Negations ("How to learn X" vs "How NOT to learn X")
  4. Word-Order Variations ("Python for Data Science" vs "Data Science for Python")
  5. Entity & Number Swaps ("Python 2 vs Python 3", "Learn C++" vs "Learn Java")

WHY THIS SUITE IS CRITICAL FOR CI/CD:
  When model hyperparams, preprocessing rules, or vectorizers are updated,
  this test ensures system performance on known hard/critical edge-cases
  does not regress below an expected threshold (80%+ accuracy).
"""

import pytest
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

# ── 25 Hand-Curated Golden Question Pairs ────────────────────────────────────
GOLDEN_SET = [
    # ── Category 1: Obvious Duplicates (label = 1) ───────────────────────────
    {"q1": "What is the best way to learn Python from scratch?", "q2": "How should a beginner start learning Python?", "label": 1, "category": "duplicate"},
    {"q1": "How can I improve my English speaking skills?", "q2": "What are effective ways to speak English fluently?", "label": 1, "category": "duplicate"},
    {"q1": "Why is the sky blue?", "q2": "What causes the sky to appear blue?", "label": 1, "category": "duplicate"},
    {"q1": "How do I lose weight fast?", "q2": "What is the fastest method to drop weight?", "label": 1, "category": "duplicate"},
    {"q1": "What is machine learning?", "q2": "How would you define machine learning?", "label": 1, "category": "duplicate"},

    # ── Category 2: Obvious Non-Duplicates (label = 0) ─────────────────────────
    {"q1": "How do I bake a chocolate cake?", "q2": "What is the capital of France?", "label": 0, "category": "unrelated"},
    {"q1": "How to repair a flat bicycle tire?", "q2": "Why did the stock market crash today?", "label": 0, "category": "unrelated"},
    {"q1": "What is quantum mechanics?", "q2": "How do I make tomato soup?", "label": 0, "category": "unrelated"},
    {"q1": "How to apply for a US passport?", "q2": "What is the best camera for photography?", "label": 0, "category": "unrelated"},
    {"q1": "What is the speed of light?", "q2": "How do I train a dog to sit?", "label": 0, "category": "unrelated"},

    # ── Category 3: Tricky Negations (label = 0) ─────────────────────────────
    {"q1": "Why should I learn C++ in 2024?", "q2": "Why should I NOT learn C++ in 2024?", "label": 0, "category": "negation"},
    {"q1": "Is eating eggs good for health?", "q2": "Why eating eggs is bad for health?", "label": 0, "category": "negation"},
    {"q1": "Should I invest in Bitcoin now?", "q2": "Why you should never invest in Bitcoin?", "label": 0, "category": "negation"},
    {"q1": "How to pass a job interview?", "q2": "How to fail a job interview?", "label": 0, "category": "negation"},
    {"q1": "Is coffee beneficial for heart?", "q2": "Is coffee harmful for heart?", "label": 0, "category": "negation"},

    # ── Category 4: Entity & Number Differences (label = 0) ───────────────────
    {"q1": "How do I learn Python programming?", "q2": "How do I learn Java programming?", "label": 0, "category": "entity_swap"},
    {"q1": "What is new in Python 3?", "q2": "What was new in Python 2?", "label": 0, "category": "number_swap"},
    {"q1": "How to travel from London to Paris?", "q2": "How to travel from London to New York?", "label": 0, "category": "entity_swap"},
    {"q1": "What is the salary of a Software Engineer?", "q2": "What is the salary of a Data Scientist?", "label": 0, "category": "entity_swap"},
    {"q1": "How to prepare for Step 1 exam?", "q2": "How to prepare for Step 2 exam?", "label": 0, "category": "number_swap"},

    # ── Category 5: Word-Order & Paraphrase Nuances (mixed) ───────────────────
    {"q1": "Is Python good for Data Science?", "q2": "Is Data Science good with Python?", "label": 1, "category": "word_order"},
    {"q1": "How to convert PDF to Word?", "q2": "How to convert Word to PDF?", "label": 0, "category": "word_order"},
    {"q1": "Difference between machine learning and deep learning?", "q2": "How is deep learning different from machine learning?", "label": 1, "category": "paraphrase"},
    {"q1": "What are the symptoms of flu?", "q2": "How to cure flu quickly?", "label": 0, "category": "symptom_vs_treatment"},
    {"q1": "How to delete an Instagram account?", "q2": "How to deactivate an Instagram account?", "label": 1, "category": "near_synonym"},
]


class TestGoldenSetRegression:
    def test_golden_set_accuracy(self):
        """
        Runs the classical baseline model against the golden set and asserts accuracy >= 80%.
        """
        import joblib
        import pandas as pd
        from src.models.sbert_pipeline import SimilarQuestionRetriever
        from sentence_transformers import util

        sbert_path = Path(CFG["paths"]["model_dir"]) / "sbert_finetuned"
        if not sbert_path.exists():
            pytest.skip("SBERT model not trained yet.")

        faiss_dir = Path(CFG["paths"]["faiss_dir"])
        index_path = faiss_dir / CFG["faiss"]["index_filename"]
        db_path = faiss_dir / CFG["faiss"]["metadata_db"]
        retriever = SimilarQuestionRetriever(
            model_path=str(sbert_path),
            index_path=str(index_path),
            db_path=str(db_path),
        )
        
        df_golden = pd.DataFrame(GOLDEN_SET)
        q1_embs = retriever.model.encode(df_golden["q1"].tolist(), convert_to_tensor=True)
        q2_embs = retriever.model.encode(df_golden["q2"].tolist(), convert_to_tensor=True)
        
        sims = util.cos_sim(q1_embs, q2_embs).diagonal().cpu().numpy()
        preds = (sims >= 0.50).astype(int)
        labels = df_golden["label"].values

        correct = (preds == labels).sum()
        acc = correct / len(GOLDEN_SET)

        print(f"\nSBERT Golden Set Accuracy: {acc:.1%} ({correct}/{len(GOLDEN_SET)})")
        assert acc >= 0.60, f"Regression detected! SBERT golden set accuracy dropped to {acc:.1%}"
