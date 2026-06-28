"""Tests for comparative retrieval evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from scholaragent.evaluation import (
    compare_retrievers,
    load_benchmark,
)
from scholaragent.retrieval import (
    BM25ScholarshipIndex,
    DenseScholarshipIndex,
    HybridScholarshipIndex,
    load_scholarships,
)

BENCHMARK = Path(
    "eval/datasets/synthetic_benchmark.json"
)
SCHOLARSHIPS = Path(
    "data/demo/synthetic_scholarships.json"
)


class KeywordEmbeddingProvider:
    """Deterministic semantic vectors for evaluation tests."""

    GROUPS = (
        (
            "artificial intelligence",
            "machine learning",
            "data science",
            "computer science",
        ),
        (
            "civil engineering",
            "structural engineering",
        ),
        (
            "public health",
            "global health",
        ),
        (
            "renewable energy",
            "sustainable energy",
        ),
        ("finland", "nordic"),
        ("germany", "european"),
        ("netherlands", "dutch"),
        ("sweden", "scandinavian"),
        ("denmark", "danish"),
    )

    def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        vectors: list[list[float]] = []

        for text in texts:
            lowered = text.casefold()

            vector = [
                float(
                    sum(
                        lowered.count(keyword)
                        for keyword in group
                    )
                )
                for group in self.GROUPS
            ]

            vector.append(1.0)
            vectors.append(vector)

        return vectors


def build_comparison():
    """Create a deterministic three-retriever comparison."""
    benchmark = load_benchmark(BENCHMARK)
    records = load_scholarships(SCHOLARSHIPS)
    embedder = KeywordEmbeddingProvider()

    bm25 = BM25ScholarshipIndex(records)
    dense = DenseScholarshipIndex(
        records,
        embedder,
    )
    hybrid = HybridScholarshipIndex(
        records,
        embedder,
    )

    return compare_retrievers(
        benchmark=benchmark,
        k=3,
        retrievers={
            "bm25": (
                lambda query, k:
                bm25.search(query, k=k)
            ),
            "dense": (
                lambda query, k:
                dense.search(query, k=k)
            ),
            "hybrid_rrf": (
                lambda query, k:
                hybrid.search(
                    query,
                    k=k,
                    candidate_k=6,
                )
            ),
        },
    )


def test_three_retrievers_are_evaluated() -> None:
    """The comparison includes all required baselines."""
    comparison = build_comparison()

    assert [
        metrics.retriever_name
        for metrics in comparison.retrievers
    ] == [
        "bm25",
        "dense",
        "hybrid_rrf",
    ]


def test_comparison_metrics_are_valid() -> None:
    """All reported metrics remain within valid bounds."""
    comparison = build_comparison()

    for metrics in comparison.retrievers:
        assert metrics.positive_cases == 5
        assert metrics.no_result_cases == 1

        assert 0 <= metrics.precision_at_k <= 1
        assert 0 <= metrics.recall_at_k <= 1
        assert 0 <= metrics.mrr <= 1
        assert 0 <= metrics.top1_hit_rate <= 1

        assert metrics.no_result_accuracy is not None
        assert 0 <= metrics.no_result_accuracy <= 1


def test_bm25_abstains_on_out_of_domain_case() -> None:
    """Lexical abstention should remain visible in comparison."""
    comparison = build_comparison()

    bm25_metrics = comparison.retrievers[0]

    assert bm25_metrics.retriever_name == "bm25"
    assert bm25_metrics.no_result_accuracy == 1.0
