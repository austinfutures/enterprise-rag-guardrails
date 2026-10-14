"""Model provider factories.

Two interchangeable tiers behind LangChain interfaces:

* **Offline** – ``HashingEmbeddings`` (signed feature hashing over stemmed unigrams +
  bigrams). Zero network, deterministic, used by CI and the default demo.
* **Hosted** – OpenAI chat + embedding models via ``langchain-openai``.

Swapping tiers is a config change (``RAG_LLM_PROVIDER`` / ``RAG_EMBEDDING_PROVIDER``),
never a code change.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings
from app.text_utils import content_terms


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-ngrams embeddings using the signed hashing trick."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _features(self, text: str) -> Counter[str]:
        terms = content_terms(text)
        feats: Counter[str] = Counter(terms)
        feats.update(f"{a}_{b}" for a, b in zip(terms, terms[1:]))
        return feats

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feat, tf in self._features(text).items():
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            weight = (1.0 + math.log(tf)) * (0.5 if "_" in feat else 1.0)
            vec[h % self.dim] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )
    return HashingEmbeddings(dim=settings.hash_embedding_dim)


def build_chat_model(settings: Settings) -> BaseChatModel | None:
    """Return a LangChain chat model, or ``None`` for the offline extractive tier."""
    if settings.llm_provider != "openai":
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=2,
        api_key=settings.openai_api_key,  # type: ignore[arg-type]
    )
