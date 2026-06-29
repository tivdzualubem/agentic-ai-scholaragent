"""Regression tests for report tables and figures."""

from __future__ import annotations

import csv
from pathlib import Path


TABLES = Path("reports/tables")
FIGURES = Path("reports/figures")


def read_csv(filename: str) -> list[dict[str, str]]:
    with (TABLES / filename).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def test_report_generators_and_dependency_exist() -> None:
    assert Path(
        "scripts/generate_report_tables.py"
    ).is_file()

    assert Path(
        "scripts/generate_report_figures.py"
    ).is_file()

    pyproject = Path(
        "pyproject.toml"
    ).read_text(encoding="utf-8")

    assert (
        'report = ["matplotlib>=3.8,<4.0"]'
        in pyproject
    )


def test_primary_retrieval_table_values() -> None:
    rows = {
        row["retriever"]: row
        for row in read_csv("retrieval_k3.csv")
    }

    assert set(rows) == {
        "bm25",
        "dense",
        "hybrid_rrf",
    }

    assert float(
        rows["bm25"]["recall_at_k"]
    ) == 1.0

    assert float(
        rows["bm25"]["mrr"]
    ) == 1.0

    assert float(
        rows["dense"]["recall_at_k"]
    ) == 0.9

    assert float(
        rows["hybrid_rrf"]["top1_hit_rate"]
    ) == 0.75


def test_rag_and_runtime_table_values() -> None:
    rag = {
        row["system"]: row
        for row in read_csv("rag_comparison.csv")
    }

    assert float(
        rag["single_pass_rag"][
            "positive_citation_pass_rate"
        ]
    ) == 0.0

    assert float(
        rag["agentic_rag"][
            "positive_citation_pass_rate"
        ]
    ) == 1.0

    assert float(
        rag["agentic_rag"][
            "positive_relevant_citation_rate"
        ]
    ) == 0.7

    runtime = {
        row["system"]: row
        for row in read_csv("runtime_and_cost.csv")
    }

    assert int(
        runtime["single_pass_rag"][
            "total_generation_calls"
        ]
    ) == 20

    assert int(
        runtime["agentic_rag"][
            "total_generation_calls"
        ]
    ) == 40


def test_all_report_figures_exist() -> None:
    expected = {
        "retrieval_k3_comparison",
        "eligibility_f1_by_status",
        "rag_quality_comparison",
        "verification_ablation",
        "mean_latency_comparison",
        "generation_call_comparison",
    }

    for stem in expected:
        for suffix in (".png", ".pdf"):
            path = FIGURES / f"{stem}{suffix}"

            assert path.is_file()
            assert path.stat().st_size > 1000

    assert len(list(FIGURES.glob("*.png"))) == 6
    assert len(list(FIGURES.glob("*.pdf"))) == 6


def test_report_interpretation_limits_are_documented() -> None:
    report = (
        TABLES / "report_tables.md"
    ).read_text(encoding="utf-8")

    readme = Path(
        "reports/README.md"
    ).read_text(encoding="utf-8")

    assert "post-hoc descriptive supplement" in report
    assert "must not be generalized" in report
    assert "deterministic verification" in report
    assert "direct hosted API fee was zero" in report

    normalized_readme = " ".join(readme.split())

    assert (
        "six official scholarship identities"
        in normalized_readme
    )
    assert "no successful citation repairs" in normalized_readme
    assert "were not monetized" in normalized_readme
