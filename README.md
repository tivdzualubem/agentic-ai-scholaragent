---
title: ScholarAgent
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
startup_duration_timeout: 1h
short_description: Verified scholarship discovery and eligibility agent
tags:
  - agentic-ai
  - rag
  - scholarships
  - streamlit
  - ollama
---

# ScholarAgent

## Live Demo

- App: https://lubem-scholaragent.hf.space
- Hugging Face Space: https://huggingface.co/spaces/lubem/scholaragent

The live app defaults to a combined 15-scholarship corpus and also allows the official, calibration, and held-out corpora to be selected separately. Evaluation results remain based on the original frozen splits.


ScholarAgent is an evidence-grounded Agentic RAG system for scholarship discovery and eligibility verification.

## Team Members

- Tivdzua Lubem Noah
- Gisele Wiykiynyuy

## Project Goal

The system will retrieve scholarship information from trusted sources, evaluate eligibility against a structured student profile, verify supporting evidence, and return cited and explainable recommendations.

## Final Technical Report

The completed project report is available here:

- [ScholarAgent Final Report](reports/ScholarAgent_Final_Report.pdf)

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

## Free Full-Stack Deployment

The Docker image includes Streamlit, BM25 retrieval, dense embeddings,
hybrid RRF retrieval, TinyLlama generation, eligibility screening,
citation verification, deterministic fallback, and safe abstention.

Build and run locally:

    docker build -t scholaragent:full-stack .
    docker run --rm -p 7860:7860 scholaragent:full-stack

Open `http://localhost:7860`.

The full Agentic RAG mode uses TinyLlama and hybrid retrieval. The fast
verified mode remains available for reliable CPU-only demonstrations.
