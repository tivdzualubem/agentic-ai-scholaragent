"""Run a live single-pass versus Agentic RAG comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from scholaragent.evaluation.defaults import (
    CALIBRATED_DENSE_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RETRIEVAL_TOP_K,
)
from scholaragent.evaluation.benchmark import (
    BenchmarkCase,
    load_benchmark,
)
from scholaragent.evaluation.rag_comparison import (
    RAGSystemName,
    evaluate_rag_comparison,
)
from scholaragent.llm.ollama_client import generate
from scholaragent.rag_baseline import TextGenerator
from scholaragent.retrieval import (
    HybridScholarshipIndex,
    OllamaEmbeddingClient,
    load_scholarships,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the live RAG-comparison parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare conventional single-pass RAG with "
            "bounded Agentic RAG."
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(
            "eval/datasets/"
            "official_development_benchmark.json"
        ),
    )
    parser.add_argument(
        "--scholarships",
        type=Path,
        default=Path(
            "data/official/"
            "official_scholarships.json"
        ),
    )
    parser.add_argument(
        "--generator-model",
        default="tinyllama:latest",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--dense-threshold",
        type=float,
        default=CALIBRATED_DENSE_THRESHOLD,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=240.0,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
    )
    parser.add_argument(
        "--max-retrieval-attempts",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser


def main() -> None:
    """Execute and display the live comparison."""
    args = build_parser().parse_args()

    benchmark = load_benchmark(args.benchmark)
    scholarships = load_scholarships(
        args.scholarships
    )

    embedder = OllamaEmbeddingClient(
        model=args.embedding_model,
        timeout=args.timeout,
    )

    index = HybridScholarshipIndex(
        scholarships,
        embedder,
        minimum_dense_score=args.dense_threshold,
    )

    def generator_factory(
        system_name: RAGSystemName,
        case: BenchmarkCase,
    ) -> TextGenerator:
        print(
            f"Running {system_name} on {case.case_id}...",
            flush=True,
        )

        def call(prompt: str) -> str:
            return generate(
                prompt,
                model=args.generator_model,
                temperature=args.temperature,
                timeout=args.timeout,
            )

        return call

    comparison = evaluate_rag_comparison(
        benchmark=benchmark,
        index=index,
        generator_factory=generator_factory,
        generator_name=args.generator_model,
        top_k=args.top_k,
        max_retrieval_attempts=(
            args.max_retrieval_attempts
        ),
        max_generation_attempts=(
            args.max_generation_attempts
        ),
        embedding_model=args.embedding_model,
        dense_threshold=args.dense_threshold,
        evaluation_scope=(
            "Small official-source development benchmark. "
            "Thresholds and results are development-only and "
            "must not be reported as final publication evidence."
        ),
    )

    print()
    print(f"Benchmark: {comparison.benchmark_name}")
    print(f"Generator: {comparison.generator_name}")
    print(f"Embedding model: {comparison.embedding_model}")
    print(
        "Dense threshold: "
        f"{comparison.dense_threshold}"
    )
    print()

    for system in comparison.systems:
        print(system.system_name)
        print(
            "  Positive completion rate:",
            system.positive_completion_rate,
        )
        print(
            "  Positive citation-pass rate:",
            system.positive_citation_pass_rate,
        )
        print(
            "  Relevant grounding rate:",
            system.positive_relevant_grounding_rate,
        )
        print(
            "  Relevant citation rate:",
            system.positive_relevant_citation_rate,
        )
        print(
            "  No-result accuracy:",
            system.no_result_accuracy,
        )
        print(
            "  Mean latency seconds:",
            round(system.mean_latency_seconds, 3),
        )
        print(
            "  Mean retrieval calls:",
            system.mean_retrieval_calls,
        )
        print(
            "  Mean generation calls:",
            system.mean_generation_calls,
        )
        print(
            "  Mean query rewrites:",
            system.mean_query_rewrites,
        )
        print(
            "  Mean repair attempts:",
            system.mean_repair_attempts,
        )
        print(
            "  Positive fallback rate:",
            system.positive_fallback_rate,
        )

        for case in system.cases:
            print(
                "   -",
                case.case_id,
                "status=",
                case.status,
                "citation=",
                case.citation_passed,
                "relevant_citation=",
                case.relevant_citation,
                "latency=",
                round(case.latency_seconds, 3),
            )

        print()

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.output.write_text(
            comparison.model_dump_json(indent=2),
            encoding="utf-8",
        )

        print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
