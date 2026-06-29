# ScholarAgent Report Artifacts

This directory contains reproducible tables and figures generated from
the tracked ScholarAgent evaluation results.

## Generate the tables

From the repository root:

    source .venv/bin/activate
    python scripts/generate_report_tables.py

## Generate the figures

Install the reporting dependency when needed:

    python -m pip install -e '.[report]'

Then run:

    python scripts/generate_report_figures.py

## Contents

`tables/` contains CSV, Markdown, and LaTeX versions of the principal
evaluation tables.

`figures/` contains matching PNG and PDF figures for retrieval,
eligibility, RAG quality, verification ablation, latency, and generation
calls.

## Evidence constraints

The primary retrieval evaluation uses Recall@3. Recall@5 is a post-hoc
descriptive supplement and was not used for model or threshold selection.

The held-out benchmark contains 24 cases over six official scholarship
identities. Perfect eligibility results therefore apply only to this
controlled benchmark and must not be generalized to arbitrary scholarship
data.

Agentic RAG completion depended on deterministic citation verification
and verified fallback. TinyLlama produced no successful citation repairs
in the held-out evaluation.

The reported zero-dollar cost represents direct hosted API fees only.
Local electricity, hardware depreciation, CPU opportunity cost, operator
time, and internet usage were not monetized.
