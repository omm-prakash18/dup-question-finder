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


# ── New Schemas ───────────────────────────────────────────────────────────────

class BatchSimilarQuestionRequest(BaseModel):
    """Request body for batch retrieval."""
    questions: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of questions to search (max 100).",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class BatchSimilarQuestionResponse(BaseModel):
    """Response for batch retrieval."""
    query_count: int
    responses: list[SimilarQuestionResponse]
    total_latency_ms: float


class IngestQuestionRequest(BaseModel):
    """Request body for dynamically adding a new question to the index."""
    question_text: str = Field(..., min_length=3, max_length=1000)

    @field_validator("question_text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question text must not be empty.")
        return v.strip()


class IngestQuestionResponse(BaseModel):
    """Response after adding a question to the index."""
    question_id: int
    question_text: str
    index_size: int
    status: str = "success"


class UserFeedbackRequest(BaseModel):
    """Request body for submitting user relevance feedback."""
    query: str
    retrieved_question_id: int
    is_relevant: bool = Field(..., description="True if retrieved question was a relevant duplicate.")
    user_rating: Optional[int] = Field(None, ge=1, le=5, description="Optional rating 1-5")


class UserFeedbackResponse(BaseModel):
    """Response confirmation for feedback submission."""
    feedback_id: int
    status: str = "logged"


class MetricsResponse(BaseModel):
    """Service operational metrics."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    index_size: int
    uptime_seconds: float

