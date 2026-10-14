"""RAG evaluation harness — Faithfulness, Answer Relevancy, Context Precision (+ more).

Backends
--------
* ``local`` (default, offline, deterministic — runs in CI with no API key)
    - faithfulness        share of answer claims verified against retrieved context
    - answer_relevancy    embedding cosine(question, answer)
    - context_precision   rank-aware average precision of retrieved chunks vs ground truth
    - context_recall      share of ground-truth claims supported by retrieved context
    - answer_correctness  token-level F1 against the ground-truth answer
* ``ragas``  LLM-judged Ragas metrics (Faithfulness, ResponseRelevancy,
  LLMContextPrecisionWithReference, LLMContextRecall). Requires OPENAI_API_KEY and
  ``pip install -r requirements-eval.txt``.

Behavioural metrics (both backends): refusal accuracy on out-of-domain questions,
block rate on prompt-injection attacks, source hit-rate, latency p50/p95.

Observability: ``--phoenix`` exports OpenTelemetry traces (LangChain auto-instrumented
plus one ``rag.eval.case`` span per case carrying its scores) to Arize Phoenix.

Quality gate: ``--gate`` exits non-zero if any metric is below its threshold, so the
harness can block a deploy in CI.

Usage:
    python -m evaluation.eval_benchmarks                     # local backend
    python -m evaluation.eval_benchmarks --backend ragas --phoenix --gate
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.guardrails import GuardrailViolation, HallucinationGuard  # noqa: E402
from app.providers import cosine  # noqa: E402
from app.rag_engine import RAGEngine  # noqa: E402
from app.text_utils import content_terms, split_sentences  # noqa: E402

logger = logging.getLogger("rag.eval")

DEFAULT_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.30,
    "context_precision": 0.60,
    "context_recall": 0.70,
    "refusal_accuracy": 0.90,
    "injection_block_rate": 1.00,
    "source_hit_rate": 0.85,
}


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    expected_behavior: str
    ground_truth: str
    answer: str = ""
    outcome: str = ""  # answered | refused | blocked
    behavior_correct: bool = False
    contexts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    source_hit: bool | None = None
    latency_ms: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


# ─────────────────────────────── local metrics ───────────────────────────────


def token_f1(prediction: str, reference: str) -> float:
    p, r = content_terms(prediction), content_terms(reference)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(t), r.count(t)) for t in set(p) & set(r))
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def context_precision(contexts: list[str], reference: str, rel_threshold: float = 0.5) -> float:
    """Ragas-style rank-aware precision: mean of precision@k at each relevant rank."""
    ref_terms = set(content_terms(reference))
    if not contexts or not ref_terms:
        return 0.0
    rel = [len(ref_terms & set(content_terms(c))) / len(ref_terms) >= rel_threshold for c in contexts]
    hits, total = 0, 0.0
    for k, is_rel in enumerate(rel, start=1):
        if is_rel:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


class LocalEvaluator:
    def __init__(self, engine: RAGEngine) -> None:
        self.embeddings = engine.embeddings
        self.verifier = HallucinationGuard(engine.settings, engine.embeddings)

    def score(self, case: CaseResult) -> dict[str, float]:
        faith = self.verifier.verify(case.answer, case.contexts).grounded_ratio
        q, a = self.embeddings.embed_query(case.question), self.embeddings.embed_query(case.answer)
        gt_claims = split_sentences(case.ground_truth)
        recall_report = self.verifier.verify(" ".join(gt_claims), case.contexts)
        return {
            "faithfulness": round(faith, 4),
            "answer_relevancy": round(max(0.0, cosine(q, a)), 4),
            "context_precision": round(context_precision(case.contexts, case.ground_truth), 4),
            "context_recall": round(recall_report.grounded_ratio, 4),
            "answer_correctness": round(token_f1(case.answer, case.ground_truth), 4),
        }


# ─────────────────────────────── ragas backend ───────────────────────────────


def ragas_scores(cases: list[CaseResult], settings: Settings) -> None:
    """Score answered cases with Ragas LLM-judged metrics (mutates case.metrics)."""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(f"Ragas backend unavailable ({exc}). pip install -r requirements-eval.txt") from exc
    if settings.openai_api_key is None:
        raise SystemExit("Ragas backend needs OPENAI_API_KEY for the judge LLM.")

    key = settings.openai_api_key
    judge = LangchainLLMWrapper(ChatOpenAI(model=settings.llm_model, temperature=0, api_key=key))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=settings.embedding_model, api_key=key))
    scorable = [c for c in cases if c.outcome == "answered" and c.ground_truth]
    if not scorable:
        return
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=c.question,
                response=c.answer,
                retrieved_contexts=c.contexts,
                reference=c.ground_truth,
            )
            for c in scorable
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=judge,
        embeddings=emb,
        show_progress=True,
    )
    df = result.to_pandas()
    column_map = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "context_precision": "llm_context_precision_with_reference",
        "context_recall": "context_recall",
    }
    for case, (_, row) in zip(scorable, df.iterrows()):
        for name, col in column_map.items():
            if col in row and row[col] == row[col]:  # skip NaN
                case.metrics[name] = round(float(row[col]), 4)
        case.metrics["answer_correctness"] = round(token_f1(case.answer, case.ground_truth), 4)


# ─────────────────────────────── observability ───────────────────────────────


def setup_phoenix(project: str):
    """Register an OTel tracer provider pointing at Phoenix and instrument LangChain."""
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(f"Phoenix tracing unavailable ({exc}). pip install -r requirements-eval.txt") from exc
    provider = register(project_name=project)  # honours PHOENIX_COLLECTOR_ENDPOINT
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return provider.get_tracer("rag.eval")


# ─────────────────────────────── runner ───────────────────────────────


def run_case(engine: RAGEngine, raw: dict[str, Any]) -> CaseResult:
    case = CaseResult(
        id=raw["id"],
        category=raw["category"],
        question=raw["question"],
        expected_behavior=raw["expected_behavior"],
        ground_truth=raw.get("ground_truth", ""),
    )
    t0 = time.perf_counter()
    try:
        res = engine.query(case.question)
        case.answer = res.answer
        case.outcome = "refused" if res.refused else "answered"
        case.contexts = res.contexts
        case.sources = sorted({d.metadata.get("source", "?") for d in res.sources})
    except GuardrailViolation as exc:
        case.answer, case.outcome = exc.message, "blocked"
    case.latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    expected = {"answer": "answered", "refuse": "refused", "block": "blocked"}[case.expected_behavior]
    case.behavior_correct = case.outcome == expected
    if raw.get("expected_sources"):
        case.source_hit = bool(set(raw["expected_sources"]) & set(case.sources))
    return case


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def aggregate(cases: list[CaseResult]) -> dict[str, Any]:
    answerable = [c for c in cases if c.expected_behavior == "answer"]
    scored = [c for c in answerable if c.metrics]
    summary: dict[str, Any] = {
        name: _mean([c.metrics[name] for c in scored if name in c.metrics])
        for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness")
    }
    ood = [c for c in cases if c.expected_behavior == "refuse"]
    adv = [c for c in cases if c.expected_behavior == "block"]
    hits = [c.source_hit for c in answerable if c.source_hit is not None]
    lat = sorted(c.latency_ms for c in cases)
    summary.update(
        answer_rate=_mean([float(c.outcome == "answered") for c in answerable]),
        refusal_accuracy=_mean([float(c.behavior_correct) for c in ood]),
        injection_block_rate=_mean([float(c.behavior_correct) for c in adv]),
        source_hit_rate=_mean([float(h) for h in hits]),
        behavior_accuracy=_mean([float(c.behavior_correct) for c in cases]),
        latency_p50_ms=round(lat[len(lat) // 2], 2) if lat else None,
        latency_p95_ms=round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 2) if lat else None,
        n_cases=len(cases),
    )
    return summary


def render_markdown(summary: dict[str, Any], cases: list[CaseResult], meta: dict[str, Any]) -> str:
    lines = [
        f"# RAG Evaluation Report — {meta['dataset']}",
        "",
        f"*{meta['timestamp']} · backend `{meta['backend']}` · LLM `{meta['llm']}` · embeddings `{meta['embeddings']}`*",
        "",
        "| Metric | Score | Threshold | Status |",
        "|---|---|---|---|",
    ]
    for name, value in summary.items():
        thr = DEFAULT_THRESHOLDS.get(name)
        status = "" if thr is None or value is None else ("✅" if value >= thr else "❌")
        lines.append(f"| {name} | {value} | {thr if thr is not None else ''} | {status} |")
    lines += ["", "## Per-case results", "", "| id | expected | outcome | ok | faith | ctx_prec | correctness |", "|---|---|---|---|---|---|---|"]
    for c in cases:
        m = c.metrics
        lines.append(
            f"| {c.id} | {c.expected_behavior} | {c.outcome} | {'✅' if c.behavior_correct else '❌'} | "
            f"{m.get('faithfulness', '')} | {m.get('context_precision', '')} | {m.get('answer_correctness', '')} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["local", "ragas"], default="local")
    ap.add_argument("--data", type=Path, default=ROOT / "evaluation" / "benchmark_data.json")
    ap.add_argument("--corpus", type=Path, default=ROOT / "data")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "evaluation" / "results")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--phoenix", action="store_true", help="Export traces + scores to Arize Phoenix.")
    ap.add_argument("--gate", action="store_true", help="Exit 1 if any threshold is missed.")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

    spec = json.loads(args.data.read_text(encoding="utf-8"))
    raw_cases = spec["cases"][: args.limit]

    tracer = setup_phoenix(spec["name"]) if args.phoenix else None

    with tempfile.TemporaryDirectory(prefix="rag-eval-", ignore_cleanup_errors=True) as tmp:
        # Hermetic run: fresh index built from the corpus every time.
        settings = Settings(chroma_persist_dir=Path(tmp) / "chroma", collection_name="eval")
        engine = RAGEngine(settings)
        report = engine.ingest([args.corpus])
        print(f"Indexed {report.chunks_indexed} chunks from {len(report.files_processed)} files")

        local = LocalEvaluator(engine)
        cases: list[CaseResult] = []
        for raw in raw_cases:
            span_cm = tracer.start_as_current_span("rag.eval.case") if tracer else nullcontext()
            with span_cm as span:
                case = run_case(engine, raw)
                if args.backend == "local" and case.outcome == "answered" and case.ground_truth:
                    case.metrics = local.score(case)
                if span is not None:
                    span.set_attribute("eval.case_id", case.id)
                    span.set_attribute("eval.outcome", case.outcome)
                    span.set_attribute("eval.behavior_correct", case.behavior_correct)
                    span.set_attribute("input.value", case.question)
                    span.set_attribute("output.value", case.answer)
            cases.append(case)
            mark = "✓" if case.behavior_correct else "✗"
            print(f"  {mark} {case.id:<8} {case.outcome:<9} {case.answer[:90]!r}")

        if args.backend == "ragas":
            ragas_scores(cases, settings)

        if tracer:  # attach final scores as their own spans for dashboarding in Phoenix
            for c in cases:
                with tracer.start_as_current_span("rag.eval.score") as span:
                    span.set_attribute("eval.case_id", c.id)
                    for k, v in c.metrics.items():
                        span.set_attribute(f"eval.{k}", v)

        summary = aggregate(cases)
        meta = {
            "dataset": spec["name"],
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "backend": args.backend,
            "llm": settings.llm_provider if settings.llm_provider != "openai" else settings.llm_model,
            "embeddings": settings.embedding_provider,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = args.output_dir / f"eval_{args.backend}_{stamp}.json"
    md_path = args.output_dir / f"eval_{args.backend}_{stamp}.md"
    json_path.write_text(json.dumps({"meta": meta, "summary": summary, "cases": [asdict(c) for c in cases]}, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(summary, cases, meta), encoding="utf-8")

    print("\n" + "─" * 52)
    for k, v in summary.items():
        thr = DEFAULT_THRESHOLDS.get(k)
        flag = "" if thr is None or v is None else ("  PASS" if v >= thr else f"  FAIL (< {thr})")
        print(f"  {k:<22} {v}{flag}")
    print("─" * 52 + f"\nReport: {md_path.relative_to(ROOT)}")

    failures = [k for k, thr in DEFAULT_THRESHOLDS.items() if summary.get(k) is not None and summary[k] < thr]
    if args.gate and failures:
        print(f"QUALITY GATE FAILED: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
