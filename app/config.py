"""Centralised, typed configuration (Pydantic Settings v2).

Every value can be overridden with an environment variable prefixed ``RAG_``
(e.g. ``RAG_LLM_PROVIDER=openai``) or via a ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── Service ──────────────────────────────────────────────────────────
    app_name: str = "Enterprise RAG & Guardrails Engine"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    api_key: SecretStr | None = Field(
        default=None, description="If set, requests must send a matching X-API-Key header."
    )

    # ── Model providers ──────────────────────────────────────────────────
    # "extractive" / "hash" run fully offline and deterministically (CI, demos).
    # "openai" uses hosted models for production-quality generation & embeddings.
    llm_provider: Literal["openai", "extractive"] = "extractive"
    embedding_provider: Literal["openai", "hash"] = "hash"
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("RAG_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_s: float = 30.0
    embedding_model: str = "text-embedding-3-small"
    hash_embedding_dim: int = Field(default=1024, ge=64)

    # ── Storage ──────────────────────────────────────────────────────────
    chroma_persist_dir: Path = Path(".chroma")
    collection_name: str = "enterprise_docs"
    data_dir: Path = Path("data")
    upload_dir: Path = Path(".uploads")
    max_upload_mb: int = Field(default=20, ge=1)
    allowed_extensions: tuple[str, ...] = (".pdf", ".txt", ".md")

    # ── Semantic chunking ────────────────────────────────────────────────
    chunk_min_chars: int = Field(default=250, ge=50)
    chunk_max_chars: int = Field(default=1200, ge=200)
    chunk_overlap_sentences: int = Field(default=1, ge=0, le=5)
    semantic_breakpoint_percentile: float = Field(
        default=25.0,
        ge=0.0,
        le=100.0,
        description="Split where adjacent-sentence similarity falls below this percentile.",
    )

    # ── Hybrid retrieval ─────────────────────────────────────────────────
    dense_k: int = Field(default=10, ge=1)
    sparse_k: int = Field(default=10, ge=1)
    final_k: int = Field(default=5, ge=1, le=20)
    dense_weight: float = Field(default=0.6, ge=0.0)
    sparse_weight: float = Field(default=0.4, ge=0.0)
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal Rank Fusion constant.")

    # ── Guardrails ───────────────────────────────────────────────────────
    max_query_chars: int = Field(default=2000, ge=10)
    injection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_dense_relevance: float | None = Field(
        default=None,
        description="Out-of-context gate on top dense similarity. None = provider default.",
    )
    min_query_coverage: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Share of query key-terms that must appear in retrieved context (0 disables).",
    )
    hallucination_mode: Literal["lexical", "llm"] = "lexical"
    claim_support_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    min_grounded_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    hallucination_action: Literal["block", "filter", "flag"] = "block"
    redact_pii: bool = True

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        needs_key = self.llm_provider == "openai" or self.embedding_provider == "openai"
        if needs_key and self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required when an 'openai' provider is selected.")
        if self.hallucination_mode == "llm" and self.llm_provider != "openai":
            raise ValueError("hallucination_mode='llm' requires llm_provider='openai'.")
        if self.chunk_min_chars >= self.chunk_max_chars:
            raise ValueError("chunk_min_chars must be smaller than chunk_max_chars.")
        return self

    @property
    def dense_relevance_floor(self) -> float:
        """Provider-calibrated similarity floor for the out-of-context check."""
        if self.min_dense_relevance is not None:
            return self.min_dense_relevance
        # Calibrated on evaluation/benchmark_data.json: offline hashing embeddings score
        # in-domain questions >= 0.128 and out-of-domain <= 0.115. Re-calibrate per corpus.
        return 0.30 if self.embedding_provider == "openai" else 0.12

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
