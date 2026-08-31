"""
api/schemas.py
─────────────────────────────────────────────────────────────────────────────
Pydantic models for FastAPI request/response validation.
Strong typing here prevents silent bugs and auto-generates OpenAPI docs.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SimilarQuestionRequest(BaseModel):
    """
    Request body for /find_similar endpoint.
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question text to find similar questions for.",
        examples=["What is the best way to learn Python?"]
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of similar questions to return (max 20).",
    )
    score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score to include in results.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be empty or whitespace only.")
        return v.strip()


class SimilarQuestion(BaseModel):
    """A single retrieved similar question with its similarity score."""
    question_id:   int
    question_text: str
    score:         float = Field(description="Cosine similarity score in [0, 1].")


class SimilarQuestionResponse(BaseModel):
    """Response from /find_similar endpoint."""
    query:           str
    results:         list[SimilarQuestion]
    result_count:    int
    latency_ms:      float = Field(description="End-to-end inference latency in ms.")


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    model_config = {"protected_namespaces": ()}

    status:          str
    model_loaded:    bool
    index_size:      int = Field(description="Number of vectors in the FAISS index.")
    device:          str
