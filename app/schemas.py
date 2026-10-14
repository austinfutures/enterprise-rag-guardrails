"""Public API contracts (Pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000, examples=["What is the hotel cap in Tier 1 cities?"])
    top_k: int | None = Field(default=None, ge=1, le=20)
    include_context: bool = Field(default=False, description="Return full retrieved passages.")


class Citation(BaseModel):
    index: int
    source: str
    page: int | None = None
    section: str | None = None
    chunk_id: str
    rrf_score: float
    dense_score: float
    sparse_score: float
    retrieved_by: str
    cited: bool
    snippet: str
    content: str | None = None


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    guardrails: dict
    latency_ms: dict[str, float]


class IngestDirectoryRequest(BaseModel):
    directory: str = Field(default=".", description="Path relative to the configured data dir.")
    force: bool = False


class IngestResponse(BaseModel):
    files_processed: list[str]
    files_skipped: list[str]
    chunks_indexed: int
    total_chunks: int
    errors: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    indexed_chunks: int
    llm_provider: str
    embedding_provider: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict = Field(default_factory=dict)
    request_id: str | None = None
