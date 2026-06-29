# ScholarAgent Demonstration

## Purpose

The Streamlit application is the working demonstration layer for
ScholarAgent. It exposes:

- structured student-profile entry;
- scholarship-query submission;
- BM25 or hybrid retrieval;
- deterministic eligibility screening;
- grounded scholarship evidence;
- citation verification;
- bounded repair and fallback;
- explicit abstention;
- agent execution traces.

The interface reuses the existing ScholarAgent workflow and does not
replace or alter the frozen evaluation pipeline.

## Installation

Run from the repository root:

    source .venv/bin/activate
    python -m pip install -e '.[demo]'

## Start the application

Run:

    ./scripts/run_demo.sh

Open:

    http://localhost:8501

## Recommended presentation configuration

Use:

- Execution mode: Fast verified demo
- Retriever mode: BM25-only offline mode

This configuration performs zero external LLM calls while demonstrating
retrieval, eligibility screening, citation-failure detection,
deterministic verified fallback, and execution tracing.

## Full Agentic RAG mode

Full mode uses:

- tinyllama:latest;
- temperature 0.0;
- deterministic citation verification;
- one bounded repair attempt;
- deterministic fallback.

It may take several minutes on CPU-only hardware.

## Hybrid retrieval mode

Hybrid mode uses:

- BM25 lexical retrieval;
- nomic-embed-text:latest;
- dense threshold 0.60;
- RRF constant 60;
- top-k 3.

It requires Ollama and the embedding model.

## Interpretation limits

The final held-out results showed:

- Agentic positive completion: 1.00;
- citation-audit pass: 1.00;
- relevant-citation rate: 0.70;
- positive fallback rate: 1.00;
- successful TinyLlama citation repairs: 0.

Therefore, safe positive completion came from bounded verification and
deterministic fallback, not successful TinyLlama citation repair.

A citation-audit pass must not be described as 100 percent
benchmark-relevant citation.
