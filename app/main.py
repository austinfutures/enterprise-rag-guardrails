"""FastAPI service exposing ingestion, guarded querying and health checks.

    POST /ingest            multipart upload of .pdf/.txt/.md files
    POST /ingest/directory  ingest a folder under the configured data dir
    POST /query             guarded hybrid-RAG question answering
    GET  /health            liveness + index status
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app import __version__
from app.config import Settings, get_settings
from app.guardrails import GuardrailViolation
from app.rag_engine import IndexEmptyError, RAGEngine
from app.schemas import (
    Citation,
    ErrorResponse,
    HealthResponse,
    IngestDirectoryRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger("rag.api")


def create_app(settings: Settings | None = None, engine: RAGEngine | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.engine = engine or RAGEngine(settings)
        logger.info("engine ready: %d chunks indexed", app.state.engine.indexed_chunks)
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )

    # ── middleware & error mapping ────────────────────────────────────────
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

    def _error(request: Request, code: int, error: str, message: str, details: dict | None = None):
        body = ErrorResponse(
            error=error,
            message=message,
            details=details or {},
            request_id=getattr(request.state, "request_id", None),
        )
        return JSONResponse(status_code=code, content=body.model_dump())

    @app.exception_handler(GuardrailViolation)
    async def _guardrail_handler(request: Request, exc: GuardrailViolation):
        logger.warning("guardrail=%s blocked request: %s", exc.guardrail, exc.details)
        return _error(request, 400, f"guardrail_violation:{exc.guardrail}", exc.message, exc.details)

    @app.exception_handler(IndexEmptyError)
    async def _empty_handler(request: Request, exc: IndexEmptyError):
        return _error(request, 503, "index_empty", str(exc))

    # ── dependencies ──────────────────────────────────────────────────────
    def get_engine(request: Request) -> RAGEngine:
        return request.app.state.engine

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = settings.api_key.get_secret_value() if settings.api_key else None
        if expected and x_api_key != expected:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")

    protected = [Depends(require_api_key)]

    # ── routes ────────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(engine: RAGEngine = Depends(get_engine)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            environment=settings.environment,
            indexed_chunks=engine.indexed_chunks,
            llm_provider=settings.llm_provider,
            embedding_provider=settings.embedding_provider,
        )

    @app.post("/ingest", response_model=IngestResponse, status_code=201, dependencies=protected, tags=["ingestion"])
    def ingest(
        files: list[UploadFile] = File(...),
        force: bool = False,
        engine: RAGEngine = Depends(get_engine),
    ) -> IngestResponse:
        if not files:
            raise HTTPException(400, "No files provided.")
        with tempfile.TemporaryDirectory(prefix="ingest-") as tmp:
            saved: list[Path] = []
            for upload in files:
                name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(upload.filename or "").name)
                suffix = Path(name).suffix.lower()
                if suffix not in settings.allowed_extensions:
                    raise HTTPException(
                        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        f"'{upload.filename}' has unsupported type '{suffix or 'none'}'. "
                        f"Allowed: {', '.join(settings.allowed_extensions)}",
                    )
                dest = Path(tmp) / name
                with dest.open("wb") as out:
                    shutil.copyfileobj(upload.file, out)
                size = dest.stat().st_size
                if size == 0:
                    raise HTTPException(400, f"'{upload.filename}' is empty.")
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, f"'{upload.filename}' exceeds {settings.max_upload_mb} MB.")
                saved.append(dest)
            report = engine.ingest(saved, force=force)
        if report.errors and not report.files_processed and not report.files_skipped:
            raise HTTPException(422, {"errors": report.errors})
        return IngestResponse(**vars(report), total_chunks=engine.indexed_chunks)

    @app.post("/ingest/directory", response_model=IngestResponse, status_code=201, dependencies=protected, tags=["ingestion"])
    def ingest_directory(
        body: IngestDirectoryRequest, engine: RAGEngine = Depends(get_engine)
    ) -> IngestResponse:
        root = settings.data_dir.resolve()
        target = (root / body.directory).resolve()
        if not target.is_relative_to(root):  # path-traversal protection
            raise HTTPException(400, "Directory must be inside the configured data dir.")
        if not target.exists():
            raise HTTPException(404, f"Directory '{body.directory}' not found.")
        report = engine.ingest([target], force=body.force)
        return IngestResponse(**vars(report), total_chunks=engine.indexed_chunks)

    @app.post("/query", response_model=QueryResponse, dependencies=protected, tags=["query"])
    def query(body: QueryRequest, engine: RAGEngine = Depends(get_engine)) -> QueryResponse:
        result = engine.query(body.question, top_k=body.top_k)
        citations = [
            Citation(
                index=i,
                source=d.metadata.get("source", "?"),
                page=d.metadata.get("page"),
                section=d.metadata.get("section"),
                chunk_id=d.metadata.get("chunk_id", "?"),
                rrf_score=d.metadata.get("rrf_score", 0.0),
                dense_score=d.metadata.get("dense_score", 0.0),
                sparse_score=d.metadata.get("sparse_score", 0.0),
                retrieved_by=d.metadata.get("retrieved_by", "?"),
                cited=i in result.cited_indices,
                snippet=d.page_content[:240],
                content=d.page_content if body.include_context else None,
            )
            for i, d in enumerate(result.sources, start=1)
        ]
        return QueryResponse(
            request_id=result.request_id,
            answer=result.answer,
            refused=result.refused,
            refusal_reason=result.refusal_reason,
            citations=citations,
            guardrails=result.guardrails,
            latency_ms=result.latency_ms,
        )

    return app


app = create_app()
