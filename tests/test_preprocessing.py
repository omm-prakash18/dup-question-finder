"""
tests/test_preprocessing.py — Comprehensive Preprocessing Bug & Edge-case Tests
─────────────────────────────────────────────────────────────────────────────
Verifies:
  1. Meaning preservation: Negations ("not", "never", "no"), numbers, question words.
  2. Edge cases: Empty strings, non-English text, emojis, punctuation only.
  3. String length extremes: Extremely short (1 word) vs extremely long (>500 words).
  4. Both modes: 'classical' (full clean) vs 'transformer' (minimal clean).
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import (
    remove_html, remove_urls, expand_contractions_text,
    clean_classical, clean_transformer
)


class TestMeaningPreservation:
    def test_preserves_negations(self):
        """CRITICAL: Negations like 'not', 'no', 'never' change sentence meaning entirely!"""
        text = "Why should I not learn C++?"
        cleaned_cls = clean_classical(text)
        cleaned_tr = clean_transformer(text)

        assert "not" in cleaned_cls, "clean_classical stripped negation 'not'!"
        assert "not" in cleaned_tr, "clean_transformer stripped negation 'not'!"

    def test_preserves_numbers(self):
        """Numbers (e.g. 'Python 3', 'top 10') carry crucial domain context."""
        text = "What is new in Python 3.12?"
        cleaned = clean_classical(text)
        assert "3" in cleaned and "12" in cleaned, "Numbers were removed during cleaning!"

    def test_preserves_question_words(self):
        """Question words ('how', 'why', 'what', 'where') distinguish intent."""
        text = "How to make coffee vs Why to drink coffee?"
        cleaned = clean_classical(text)
        assert "how" in cleaned and "why" in cleaned


class TestEdgeCases:
    def test_empty_and_whitespace(self):
        assert clean_classical("") == ""
        assert clean_classical("   \n\t  ") == ""
        assert clean_transformer("") == ""

    def test_punctuation_and_emojis_only(self):
        text = "??? !!! 😀 🚀 ***"
        cleaned_cls = clean_classical(text)
        cleaned_tr = clean_transformer(text)
        # Should not crash
        assert isinstance(cleaned_cls, str)
        assert isinstance(cleaned_tr, str)

    def test_html_entities(self):
        text = "What is R&amp;D in tech &gt; finance?"
        cleaned_cls = clean_classical(text)
        assert "&amp;" not in cleaned_cls
        assert "r d" in cleaned_cls or "r" in cleaned_cls

    def test_urls_and_links(self):
        text = "Check http://example.com/page?id=1 or www.test.com"
        cleaned_tr = clean_transformer(text)
        assert "<URL>" in cleaned_tr
        assert "http://" not in cleaned_tr

    def test_extreme_lengths(self):
        short_q = "Why?"
        long_q = "word " * 1000

        assert len(clean_classical(short_q)) > 0
        assert len(clean_classical(long_q)) > 0
        assert len(clean_transformer(long_q)) > 0

    def test_non_english(self):
        text = "¿Cómo aprender Python en 2024?"
        cleaned_cls = clean_classical(text)
        cleaned_tr = clean_transformer(text)
        assert "cómo" in cleaned_cls.lower() or "aprender" in cleaned_cls.lower()
        assert len(cleaned_tr) > 0
