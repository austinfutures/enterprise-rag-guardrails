"""Document intelligence & vector ingestion pipeline.

load → clean → structure (sections) → semantic chunk → enrich metadata → upsert (Chroma)

* **Loaders**: PDF (per-page, via ``pypdf``), Markdown and plain text.
* **Cleaning**: Unicode normalisation, de-hyphenation across line breaks,
  whitespace collapsing, control-character removal.
* **Structure**: Markdown headings / ALL-CAPS / numbered headings start new sections;
  chunks never straddle a section boundary and carry the section title as metadata.
* **Semantic chunking**: sentences are embedded; a chunk boundary is placed where
  adjacent-sentence similarity drops below a percentile of that section's own similarity
  distribution (so the threshold adapts per document), bounded by min/max chunk sizes,
  with sentence-level overlap.
* **Idempotent upserts**: chunk IDs are content-addressed (``<doc_hash>:p<page>:c<n>``);
  re-ingesting an unchanged file is a no-op and a changed file replaces its old chunks.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.text_utils import normalize_unicode, split_sentences

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Base class for ingestion failures."""


class UnsupportedFileTypeError(IngestionError):
    pass


class DocumentParseError(IngestionError):
    pass


@dataclass
class PageText:
    text: str
    page: int


@dataclass
class IngestionReport:
    files_processed: list[str] = field(default_factory=list)
    files_skipped: list[str] = field(default_factory=list)
    chunks_indexed: int = 0
    errors: dict[str, str] = field(default_factory=dict)


# ─────────────────────────────── Loading & cleaning ───────────────────────────────


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate wrapped words
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pages(path: Path) -> list[PageText]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                raise DocumentParseError(f"{path.name}: encrypted PDFs are not supported")
            pages = [
                PageText(clean_text(p.extract_text() or ""), i + 1)
                for i, p in enumerate(reader.pages)
            ]
        except (PdfReadError, ValueError, KeyError) as exc:
            raise DocumentParseError(f"{path.name}: unreadable PDF ({exc})") from exc
    elif suffix in {".txt", ".md"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        # Form-feeds denote page breaks in exported text documents.
        pages = [PageText(clean_text(p), i + 1) for i, p in enumerate(raw.split("\f"))]
    else:
        raise UnsupportedFileTypeError(f"{path.name}: unsupported file type '{suffix}'")

    pages = [p for p in pages if p.text]
    if not pages:
        raise DocumentParseError(f"{path.name}: no extractable text (scanned PDF needs OCR)")
    return pages


_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+.+|(?:\d+(?:\.\d+)*\.?\s+)[A-Z][A-Za-z &/,()'-]{2,80}|[A-Z][A-Z0-9 &/,()'-]{3,80})$"
)


def split_sections(text: str, default_title: str) -> list[tuple[str, str]]:
    """Split page text into (section_title, body) pairs using heading heuristics."""
    sections: list[tuple[str, str]] = []
    title, buf = default_title, []
    for line in text.splitlines():
        stripped = line.strip()
        is_heading = bool(_HEADING_RE.match(stripped)) and not stripped.endswith(".")
        if is_heading and len(stripped.split()) <= 12:
            if "\n".join(buf).strip():
                sections.append((title, "\n".join(buf).strip()))
            title, buf = stripped.lstrip("# ").strip(), []
        else:
            buf.append(line)
    if "\n".join(buf).strip():
        sections.append((title, "\n".join(buf).strip()))
    return sections


def document_title(path: Path, pages: list[PageText]) -> str:
    for line in pages[0].text.splitlines():
        if line.strip():
            return line.strip().lstrip("# ").strip()[:120]
    return path.stem


# ─────────────────────────────── Semantic chunking ───────────────────────────────


class SemanticChunker:
    """Embedding-driven, size-bounded, overlap-aware chunker."""

    def __init__(
        self,
        embeddings: Embeddings,
        min_chars: int = 250,
        max_chars: int = 1200,
        breakpoint_percentile: float = 25.0,
        overlap_sentences: int = 1,
    ) -> None:
        self.embeddings = embeddings
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.breakpoint_percentile = breakpoint_percentile
        self.overlap_sentences = overlap_sentences
        self._fallback = RecursiveCharacterTextSplitter(
            chunk_size=max_chars, chunk_overlap=min(150, max_chars // 8)
        )

    def _sentences(self, text: str) -> list[str]:
        out: list[str] = []
        for s in split_sentences(text):
            out.extend(self._fallback.split_text(s) if len(s) > self.max_chars else [s])
        return out

    def _breakpoints(self, sentences: list[str]) -> set[int]:
        """Indices *i* where a new chunk should start before sentence *i*."""
        if len(sentences) < 3:
            return set()
        # Embed each sentence with a 1-sentence window either side to smooth noise.
        windows = [" ".join(sentences[max(0, i - 1) : i + 2]) for i in range(len(sentences))]
        vecs = np.asarray(self.embeddings.embed_documents(windows), dtype=float)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
        threshold = float(np.percentile(sims, self.breakpoint_percentile))
        return {i + 1 for i, s in enumerate(sims) if s < threshold}

    def split(self, text: str) -> list[str]:
        sentences = self._sentences(text)
        if not sentences:
            return []
        breaks = self._breakpoints(sentences)
        chunks: list[list[str]] = []
        current: list[str] = []
        for i, sent in enumerate(sentences):
            cur_len = sum(len(s) + 1 for s in current)
            semantic_break = i in breaks and cur_len >= self.min_chars
            size_break = current and cur_len + len(sent) > self.max_chars
            if current and (semantic_break or size_break):
                chunks.append(current)
                current = current[-self.overlap_sentences :] if self.overlap_sentences else []
                # Never let overlap alone push the next chunk past the max size.
                while current and sum(len(s) + 1 for s in current) + len(sent) > self.max_chars:
                    current.pop(0)
            current.append(sent)
        if current:
            # Merge a tiny trailing chunk into its predecessor when it fits.
            tail_len = sum(len(s) + 1 for s in current)
            if chunks and tail_len < self.min_chars // 2:
                merged = chunks[-1] + [s for s in current if s not in chunks[-1]]
                if sum(len(s) + 1 for s in merged) <= self.max_chars:
                    chunks[-1] = merged
                    current = []
            if current:
                chunks.append(current)
        return [" ".join(c).strip() for c in chunks if " ".join(c).strip()]


# ─────────────────────────────── Pipeline ───────────────────────────────


class DocumentIngestor:
    """Turns files into enriched chunks and upserts them into a LangChain vector store."""

    def __init__(self, vectorstore, chunker: SemanticChunker) -> None:
        self.vectorstore = vectorstore
        self.chunker = chunker

    def build_chunks(self, path: Path) -> list[Document]:
        pages = load_pages(path)
        digest = file_sha256(path)
        doc_id = digest[:16]
        title = document_title(path, pages)
        ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        chunks: list[Document] = []
        for page in pages:
            for section, body in split_sections(page.text, default_title=title):
                for text in self.chunker.split(body):
                    idx = len(chunks)
                    chunks.append(
                        Document(
                            page_content=text,
                            metadata={
                                "chunk_id": f"{doc_id}:p{page.page}:c{idx}",
                                "doc_id": doc_id,
                                "source": path.name,
                                "file_type": path.suffix.lower().lstrip("."),
                                "title": title,
                                "section": section[:200],
                                "page": page.page,
                                "chunk_index": idx,
                                "char_count": len(text),
                                "content_sha256": digest,
                                "ingested_at": ingested_at,
                            },
                        )
                    )
        return chunks

    def _existing_hashes(self, source: str) -> tuple[list[str], set[str]]:
        found = self.vectorstore.get(where={"source": source}, include=["metadatas"])
        hashes = {m.get("content_sha256", "") for m in found.get("metadatas") or []}
        return found.get("ids") or [], hashes

    def ingest_file(self, path: Path, report: IngestionReport, force: bool = False) -> None:
        old_ids, old_hashes = self._existing_hashes(path.name)
        digest = file_sha256(path)
        if not force and old_hashes == {digest}:
            logger.info("skip unchanged file %s", path.name)
            report.files_skipped.append(path.name)
            return
        chunks = self.build_chunks(path)
        if old_ids:  # replace stale version of the same document
            self.vectorstore.delete(ids=old_ids)
        self.vectorstore.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
        report.files_processed.append(path.name)
        report.chunks_indexed += len(chunks)
        logger.info("indexed %s → %d chunks", path.name, len(chunks))

    def ingest_paths(
        self, paths: list[Path], allowed: tuple[str, ...], force: bool = False
    ) -> IngestionReport:
        report = IngestionReport()
        for path in paths:
            if path.is_dir():
                files = sorted(p for p in path.rglob("*") if p.suffix.lower() in allowed)
            else:
                files = [path]
            for f in files:
                if f.suffix.lower() not in allowed:
                    report.errors[f.name] = f"unsupported file type '{f.suffix}'"
                    continue
                try:
                    self.ingest_file(f, report, force=force)
                except IngestionError as exc:
                    report.errors[f.name] = str(exc)
                    logger.warning("ingestion failed: %s", exc)
        return report
