# ScholarAgent

ScholarAgent is an evidence-grounded Agentic RAG system for scholarship discovery and eligibility verification.

## Team Members

- Tivdzua Lubem Noah
- Gisele Wiykiynyuy

## Project Goal

The system will retrieve scholarship information from trusted sources, evaluate eligibility against a structured student profile, verify supporting evidence, and return cited and explainable recommendations.

## Working demonstration UI

ScholarAgent includes a Streamlit application for:

- scholarship-query submission;
- structured student-profile entry;
- BM25 or hybrid retrieval;
- eligibility screening;
- official evidence inspection;
- citation verification;
- fallback recovery;
- abstention;
- agent execution traces.

Install the optional demonstration dependency:

    source .venv/bin/activate
    python -m pip install -e '.[demo]'

Start the application:

    ./scripts/run_demo.sh

Then open:

    http://localhost:8501

For the most reliable live presentation, use:

- Execution mode: Fast verified demo
- Retriever mode: BM25-only offline mode

This configuration performs zero external LLM calls while demonstrating
the verification and deterministic fallback pathway.

Hybrid retrieval requires Ollama with nomic-embed-text:latest. Full
Agentic RAG additionally requires tinyllama:latest and may be slow on
CPU-only hardware.

See docs/demo.md for the complete demonstration protocol and
interpretation limits.
