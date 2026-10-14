"""Guardrails: defence-in-depth around the RAG pipeline.

Layer                      Threat                                   Response
─────────────────────────  ───────────────────────────────────────  ──────────────────────
1. Input sanitisation      control chars, Unicode smuggling, size   normalise / reject 400
2. Prompt-injection guard  direct jailbreaks & instruction override reject 400
3. Context quarantine      *indirect* injection inside documents    drop poisoned chunks
4. Out-of-context gate     questions the corpus cannot answer       grounded refusal
5. Hallucination guard     answer claims not entailed by context    block / filter / flag
6. PII redaction           sensitive data leaking in the output     mask
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from app.config import Settings
from app.providers import cosine
from app.text_utils import (
    content_terms,
    extract_numbers,
    normalize_unicode,
    split_sentences,
    strip_citations,
)

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = (
    "I don't have enough information in the provided documents to answer that question."
)
HALLUCINATION_FALLBACK = (
    "I couldn't produce an answer that is fully supported by the source documents. "
    "Please rephrase the question or consult the cited documents directly."
)


class GuardrailViolation(Exception):
    """Raised when a request must be rejected outright (HTTP 400)."""

    def __init__(self, guardrail: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.guardrail = guardrail
        self.message = message
        self.details = details or {}


# ─────────────────────────────── 1+2. Input guard ───────────────────────────────

# (pattern, weight, label). Scores combine with a noisy-OR so several weak signals
# can add up to a block while one strong signal is enough on its own.
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], float, str]] = [
    (re.compile(p, re.I), w, label)
    for p, w, label in [
        (r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|the)\b.{0,30}\b(instructions?|rules?|prompts?|guidelines?|context|directions?)", 0.9, "instruction_override"),
        (r"\b(reveal|show|print|display|repeat|output|leak|tell me)\b.{0,40}\b(system|hidden|initial|original|developer)\s+(prompt|instructions?|message)", 0.9, "prompt_extraction"),
        (r"\byou are (now|no longer)\b|\bfrom now on,? you\b", 0.6, "role_reassignment"),
        (r"\b(act|behave|respond)\s+as\s+(an?\s+)?(unrestricted|unfiltered|jailbroken|different|evil)\b", 0.7, "role_play_jailbreak"),
        (r"\bpretend (to be|you are|that)\b", 0.4, "pretend"),
        (r"\b(developer|god|admin|debug|sudo)\s+mode\b", 0.7, "privileged_mode"),
        (r"\b(DAN|do anything now|jailbreak|jail-break)\b", 0.8, "jailbreak_keyword"),
        (r"<\|?(im_start|im_end|system|endoftext)\|?>|\[/?(INST|SYS)\]|<<SYS>>", 0.9, "special_tokens"),
        (r"(^|\n)\s*(#{2,}\s*)?(system|assistant)\s*:", 0.6, "fake_role_header"),
        (r"\b(answer|respond)\b.{0,30}\b(without|regardless of|outside)\b.{0,20}\b(context|documents?|sources?|restrictions?)", 0.6, "context_escape"),
        (r"\buse your (own|general|prior) knowledge\b", 0.5, "context_escape"),
        (r"\b(new|updated|real)\s+instructions?\s*:", 0.6, "instruction_injection"),
        (r"\b(exfiltrate|send (this|the data) to|curl\s+http|fetch\s+http)", 0.6, "exfiltration"),
        (r"[A-Za-z0-9+/]{120,}={0,2}", 0.4, "encoded_payload"),
    ]
]


@dataclass
class InputCheck:
    sanitized: str
    injection_score: float
    matched_patterns: list[str]
    passed: bool


def injection_score(text: str) -> tuple[float, list[str]]:
    matched: list[str] = []
    survival = 1.0
    for pattern, weight, label in _INJECTION_PATTERNS:
        if pattern.search(text):
            matched.append(label)
            survival *= 1.0 - weight
    return round(1.0 - survival, 4), matched


class InputGuard:
    def __init__(self, settings: Settings) -> None:
        self.max_chars = settings.max_query_chars
        self.threshold = settings.injection_threshold

    def check(self, query: str) -> InputCheck:
        sanitized = re.sub(r"\s+", " ", normalize_unicode(query)).strip()
        if not sanitized:
            raise GuardrailViolation("input_validation", "Query is empty after sanitisation.")
        if len(sanitized) > self.max_chars:
            raise GuardrailViolation(
                "input_validation",
                f"Query exceeds {self.max_chars} characters.",
                {"length": len(sanitized)},
            )
        score, matched = injection_score(sanitized)
        if score >= self.threshold:
            raise GuardrailViolation(
                "prompt_injection",
                "Request blocked: potential prompt-injection attempt detected.",
                {"injection_score": score, "matched_patterns": matched},
            )
        return InputCheck(sanitized, score, matched, True)


# ─────────────────────────────── 3+4. Context guards ───────────────────────────────


def quarantine_chunks(
    docs: Sequence[Document], threshold: float
) -> tuple[list[Document], list[str]]:
    """Drop retrieved chunks carrying embedded instructions (indirect prompt injection)."""
    safe, dropped = [], []
    for d in docs:
        score, _ = injection_score(d.page_content)
        if score >= threshold:
            dropped.append(d.metadata.get("chunk_id", "?"))
        else:
            safe.append(d)
    return safe, dropped


@dataclass
class ContextCheck:
    in_context: bool
    top_dense_relevance: float
    query_coverage: float
    reason: str | None = None


def query_coverage(query: str, docs: Sequence[Document]) -> float:
    q_terms = set(content_terms(query))
    if not q_terms:
        return 0.0
    ctx_terms = set(content_terms(" ".join(d.page_content for d in docs)))
    return round(len(q_terms & ctx_terms) / len(q_terms), 4)


def check_context(
    query: str, docs: Sequence[Document], top_dense: float, settings: Settings
) -> ContextCheck:
    coverage = query_coverage(query, docs)
    if not docs:
        return ContextCheck(False, top_dense, coverage, "no_documents_retrieved")
    # Numeric anchors (years, fiscal periods, amounts) in the question must exist in the
    # context; otherwise "revenue in FY2019" would be answered with FY2025 figures.
    ctx_terms = set(content_terms(" ".join(d.page_content for d in docs)))
    anchors = {t for t in content_terms(query) if any(c.isdigit() for c in t)}
    if anchors - ctx_terms:
        return ContextCheck(False, top_dense, coverage, "missing_numeric_anchor")
    if top_dense < settings.dense_relevance_floor:
        return ContextCheck(False, top_dense, coverage, "low_semantic_relevance")
    if settings.min_query_coverage and coverage < settings.min_query_coverage:
        return ContextCheck(False, top_dense, coverage, "low_lexical_coverage")
    return ContextCheck(True, top_dense, coverage)


# ─────────────────────────────── 5. Hallucination guard ───────────────────────────────


@dataclass
class ClaimVerdict:
    claim: str
    support: float
    supported: bool
    reason: str = ""


@dataclass
class GroundingReport:
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def grounded_ratio(self) -> float:
        if not self.verdicts:
            return 1.0
        return round(sum(v.supported for v in self.verdicts) / len(self.verdicts), 4)

    @property
    def unsupported(self) -> list[str]:
        return [v.claim for v in self.verdicts if not v.supported]


class _LLMVerdicts(BaseModel):
    supported: list[bool] = Field(description="One boolean per claim, in order.")


class HallucinationGuard:
    """Verifies each generated statement against the retrieved context.

    *lexical* mode (default, offline): a claim is supported when enough of its key terms
    occur in a single context passage (or it is semantically close to a context sentence)
    **and** every number it states literally appears in the context — numeric fabrication
    is the most damaging hallucination in financial / policy domains.

    *llm* mode: an LLM-as-judge entailment check with structured output.
    """

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.settings = settings
        self.threshold = settings.claim_support_threshold
        self.embeddings = embeddings
        self.llm = llm

    @staticmethod
    def claims(answer: str) -> list[str]:
        return [
            s
            for s in (strip_citations(x).strip() for x in split_sentences(answer))
            if len(content_terms(s)) >= 2
        ]

    def verify(self, answer: str, contexts: Sequence[str]) -> GroundingReport:
        claims = self.claims(answer)
        if not claims:
            return GroundingReport()
        if self.settings.hallucination_mode == "llm" and self.llm is not None:
            return self._verify_llm(claims, contexts)
        return self._verify_lexical(claims, contexts)

    def _verify_lexical(self, claims: list[str], contexts: Sequence[str]) -> GroundingReport:
        ctx_term_sets = [set(content_terms(c)) for c in contexts]
        ctx_numbers = set().union(*(extract_numbers(c) for c in contexts)) if contexts else set()
        ctx_sentences = [s for c in contexts for s in split_sentences(c)]
        ctx_vecs = (
            self.embeddings.embed_documents(ctx_sentences)
            if self.embeddings is not None and ctx_sentences
            else []
        )
        report = GroundingReport()
        for claim in claims:
            terms = set(content_terms(claim))
            lexical = max((len(terms & s) / len(terms) for s in ctx_term_sets), default=0.0)
            semantic = 0.0
            if ctx_vecs:
                cv = self.embeddings.embed_query(claim)  # type: ignore[union-attr]
                semantic = max(cosine(cv, v) for v in ctx_vecs)
            support = round(max(lexical, semantic), 4)
            missing_numbers = extract_numbers(claim) - ctx_numbers
            if missing_numbers:
                report.verdicts.append(
                    ClaimVerdict(claim, 0.0, False, f"numbers not in context: {sorted(missing_numbers)}")
                )
            else:
                ok = support >= self.threshold
                report.verdicts.append(
                    ClaimVerdict(claim, support, ok, "" if ok else "insufficient textual support")
                )
        return report

    def _verify_llm(self, claims: list[str], contexts: Sequence[str]) -> GroundingReport:
        prompt = (
            "You are a strict fact-checker. For each CLAIM decide whether it is fully "
            "entailed by the CONTEXT alone (no outside knowledge). Numbers must match exactly.\n\n"
            f"CONTEXT:\n{chr(10).join(contexts)}\n\nCLAIMS:\n"
            + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        )
        try:
            judge = self.llm.with_structured_output(_LLMVerdicts)  # type: ignore[union-attr]
            result: _LLMVerdicts = judge.invoke(prompt)  # type: ignore[assignment]
            flags = (result.supported + [False] * len(claims))[: len(claims)]
        except Exception as exc:  # fail closed, fall back to lexical verification
            logger.warning("LLM judge failed (%s); falling back to lexical check", exc)
            return self._verify_lexical(claims, contexts)
        return GroundingReport(
            [ClaimVerdict(c, float(f), f, "" if f else "judge: not entailed") for c, f in zip(claims, flags)]
        )


# ─────────────────────────────── 6. PII redaction ───────────────────────────────

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")),
]


def _luhn_ok(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 16:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for label, pattern in _PII_PATTERNS:

        def _sub(m: re.Match[str], label: str = label) -> str:
            if label == "CREDIT_CARD" and not _luhn_ok(m.group()):
                return m.group()
            counts[label] = counts.get(label, 0) + 1
            return f"[REDACTED_{label}]"

        text = pattern.sub(_sub, text)
    return text, counts


def is_refusal(answer: str) -> bool:
    return answer.strip().lower().startswith(REFUSAL_MESSAGE[:45].lower())


__all__ = [
    "REFUSAL_MESSAGE",
    "HALLUCINATION_FALLBACK",
    "GuardrailViolation",
    "InputGuard",
    "InputCheck",
    "ContextCheck",
    "check_context",
    "quarantine_chunks",
    "query_coverage",
    "HallucinationGuard",
    "GroundingReport",
    "redact_pii",
    "injection_score",
    "is_refusal",
]
