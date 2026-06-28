"""Compare BM25, dense, and hybrid scholarship retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from scholaragent.evaluation import load_benchmark
from scholaragent.evaluation.retrieval_comparison import (
    compare_retrievers,
)
from scholaragent.retrieval import (
    BM25ScholarshipIndex,
    DenseScholarshipIndex,
    HybridScholarshipIndex,
    OllamaEmbeddingClient,
    load_scholarships,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the retrieval-comparison CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare BM25, dense, and hybrid retrieval."
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(
            "eval/datasets/synthetic_benchmark.json"
        ),
    )
    parser.add_argument(
        "--scholarships",
        type=Path,
        default=Path(
            "data/demo/synthetic_scholarships.json"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--dense-threshold",
        type=float,
        default=0.67,
        help=(
            "Minimum cosine similarity for dense evidence. "
            "The default is a synthetic development value."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser


def main() -> None:
    """Run and display the retrieval comparison."""
    args = build_parser().parse_args()

    benchmark = load_benchmark(args.benchmark)
    scholarships = load_scholarships(
        args.scholarships
    )

    embedder = OllamaEmbeddingClient(
        model="nomic-embed-text",
    )

    bm25 = BM25ScholarshipIndex(scholarships)
    dense = DenseScholarshipIndex(
        scholarships,
        embedder,
    )
    hybrid = HybridScholarshipIndex(
        scholarships,
        embedder,
        minimum_dense_score=args.dense_threshold,
    )

    comparison = compare_retrievers(
        benchmark=benchmark,
        k=args.top_k,
        retrievers={
            "bm25": (
                lambda query, k:
                bm25.search(query, k=k)
            ),
            "dense": (
                lambda query, k:
                dense.search(
                    query,
                    k=k,
                    minimum_score=args.dense_threshold,
                )
            ),
            "hybrid_rrf": (
                lambda query, k:
                hybrid.search(
                    query,
                    k=k,
                    candidate_k=max(k * 3, k),
                )
            ),
        },
    )

    print(f"Benchmark: {comparison.benchmark_name}")
    print(f"Top-k: {comparison.k}")
    print(
        "Dense abstention threshold: "
        f"{args.dense_threshold:.6f}"
    )
    print()
    print(
        f"{'Retriever':<14}"
        f"{'P@K':>10}"
        f"{'R@K':>10}"
        f"{'MRR':>10}"
        f"{'Top-1':>10}"
        f"{'No-result':>12}"
    )
    print("-" * 66)

    for metrics in comparison.retrievers:
        no_result = (
            f"{metrics.no_result_accuracy:.4f}"
            if metrics.no_result_accuracy is not None
            else "n/a"
        )

        print(
            f"{metrics.retriever_name:<14}"
            f"{metrics.precision_at_k:>10.4f}"
            f"{metrics.recall_at_k:>10.4f}"
            f"{metrics.mrr:>10.4f}"
            f"{metrics.top1_hit_rate:>10.4f}"
            f"{no_result:>12}"
        )

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.output.write_text(
            comparison.model_dump_json(indent=2),
            encoding="utf-8",
        )
        print()
        print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
