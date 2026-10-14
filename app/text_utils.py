"""Shared, dependency-free text primitives used across ingestion, retrieval,
guardrails and evaluation so every component tokenises identically."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS: frozenset[str] = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just me more most my myself no nor not now of off on once
    only or other our ours ourselves out over own same she should so some such than that the
    their theirs them themselves then there these they this those through to too under until
    up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves many much per may must also us tell give please according
    within without whats does doesnt dont""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9$])")
_BULLET_RE = re.compile(r"^\s*(?:[-*•▪]|\d+[.)])\s+")


def normalize_unicode(text: str) -> str:
    """NFKC-normalise and strip zero-width / control characters (keeps \\n and \\t)."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in {"Cc", "Cf"}
    )


def stem(token: str) -> str:
    """Very light, deterministic suffix stripping (good enough for lexical matching)."""
    if token.isdigit() or len(token) <= 4:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 6:
        return token[:-3]
    if token.endswith("ed") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-cased word/number tokens, numbers keep decimals (``4.82``)."""
    return [t.replace(",", "") for t in _TOKEN_RE.findall(text.lower())]


def content_terms(text: str) -> list[str]:
    """Stemmed, stop-word-filtered tokens (order preserved, duplicates kept)."""
    return [stem(t) for t in tokenize(strip_citations(text)) if t not in STOPWORDS and len(t) > 1]


def extract_numbers(text: str) -> set[str]:
    """Numeric literals, comma-insensitive, trailing zeros on decimals kept as-is."""
    return {n.replace(",", "") for n in _NUMBER_RE.findall(strip_citations(text))}


def strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text)


def find_citations(text: str) -> list[int]:
    out: list[int] = []
    for group in _CITATION_RE.findall(text):
        out.extend(int(x) for x in group.split(","))
    return out


def split_sentences(text: str) -> list[str]:
    """Paragraph-, bullet- and punctuation-aware sentence splitter."""
    units: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            units.append(" ".join(paragraph))
            paragraph.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif _BULLET_RE.match(line) or line.startswith("#"):
            flush()
            units.append(_BULLET_RE.sub("", line).lstrip("# ").strip())
        else:
            paragraph.append(line)
    flush()

    sentences: list[str] = []
    for unit in units:
        unit = re.sub(r"\s+", " ", unit).strip()
        sentences.extend(s.strip() for s in _SENT_SPLIT_RE.split(unit) if s.strip())
    return sentences
