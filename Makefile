PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python
HOST   ?= 0.0.0.0
PORT   ?= 8000

.DEFAULT_GOAL := help
.PHONY: help setup setup-eval run ingest query test evaluate evaluate-ragas phoenix clean

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install pinned core dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@test -f .env || cp .env.example .env

setup-eval: setup  ## Also install Ragas + Phoenix tracing extras
	$(BIN)/pip install -r requirements-eval.txt

run:  ## Start the API with hot reload (docs at http://localhost:8000/docs)
	$(BIN)/uvicorn app.main:app --host $(HOST) --port $(PORT) --reload

ingest:  ## Ingest everything in ./data into the persistent index (no server needed)
	$(PY) -c "from app.config import get_settings; from app.rag_engine import RAGEngine; s=get_settings(); print(RAGEngine(s).ingest([s.data_dir]))"

query:  ## Example query against a running server: make query Q="What is the meal per diem?"
	curl -s -X POST localhost:$(PORT)/query -H 'Content-Type: application/json' \
	  -d '{"question": "$(or $(Q),What is the meal per diem?)"}' | $(PY) -m json.tool

test:  ## Run the pytest suite (offline, deterministic)
	$(PY) -m pytest

evaluate:  ## Offline benchmark (faithfulness, relevancy, context precision) + quality gate
	$(PY) -m evaluation.eval_benchmarks --backend local --gate

evaluate-ragas:  ## LLM-judged Ragas benchmark with Phoenix tracing (needs OPENAI_API_KEY)
	$(PY) -m evaluation.eval_benchmarks --backend ragas --phoenix --gate

phoenix:  ## Launch the Arize Phoenix UI locally via Docker (http://localhost:6006)
	docker run --rm -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest

clean:  ## Remove index, uploads, caches and eval results
	rm -rf .chroma .uploads .pytest_cache evaluation/results
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
