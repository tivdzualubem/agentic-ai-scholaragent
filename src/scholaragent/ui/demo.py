"""Reusable workflow logic for the ScholarAgent demo application."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from scholaragent.agentic_rag import (
    AgenticRAGResult,
    run_agentic_rag,
)
from scholaragent.llm.ollama_client import generate
from scholaragent.retrieval import (
    BM25ScholarshipIndex,
    HybridScholarshipIndex,
    OllamaEmbeddingClient,
    ScholarshipSearchIndex,
    load_scholarships,
)
from scholaragent.schemas import StudentProfile


DEFAULT_GENERATOR_MODEL = "tinyllama:latest"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
DEFAULT_DENSE_THRESHOLD = 0.60
DEFAULT_RRF_CONSTANT = 60
DEFAULT_TOP_K = 3
DEFAULT_TIMEOUT_SECONDS = 900.0


class DemoExecutionMode(StrEnum):
    """Execution modes available in the demonstration."""

    FAST_VERIFIED = "fast_verified"
    FULL_AGENTIC = "full_agentic"


class DemoRetrieverMode(StrEnum):
    """Retrieval strategies available in the demonstration."""

    HYBRID_RRF = "hybrid_rrf"
    BM25_ONLY = "bm25_only"


class DemoExecution(BaseModel):
    """One completed demonstration execution."""

    model_config = ConfigDict(extra="forbid")

    execution_mode: DemoExecutionMode
    retriever_mode: DemoRetrieverMode
    corpus_path: str
    corpus_size: int = Field(ge=1)
    external_llm_calls: int = Field(ge=0)
    result: AgenticRAGResult


def parse_csv_values(raw_value: str) -> list[str]:
    """Parse comma-separated text into normalized unique values."""
    values: list[str] = []
    seen: set[str] = set()

    for item in raw_value.split(","):
        normalized = item.strip()

        if not normalized:
            continue

        key = normalized.casefold()

        if key in seen:
            continue

        values.append(normalized)
        seen.add(key)

    return values


def build_student_profile(
    *,
    nationality: str,
    country_of_residence: str,
    target_degree_level: str,
    fields_of_study: str,
    include_gpa: bool,
    gpa: float,
    gpa_scale: float,
    include_language_score: bool,
    language_test: str,
    language_score: float,
    years_work_experience: float,
    preferred_countries: str,
    requires_full_funding: bool,
) -> StudentProfile:
    """Convert form-compatible values into a validated profile."""
    language_scores: dict[str, float] = {}

    if include_language_score:
        normalized_test = language_test.strip()

        if not normalized_test:
            raise ValueError(
                "Select a language test or disable the "
                "language-score option."
            )

        language_scores[normalized_test] = language_score

    return StudentProfile(
        nationality=nationality,
        country_of_residence=(
            country_of_residence.strip() or None
        ),
        target_degree_level=target_degree_level,
        fields_of_study=parse_csv_values(
            fields_of_study
        ),
        gpa=gpa if include_gpa else None,
        gpa_scale=gpa_scale if include_gpa else None,
        language_scores=language_scores,
        years_work_experience=years_work_experience,
        preferred_countries=parse_csv_values(
            preferred_countries
        ),
        requires_full_funding=requires_full_funding,
    )


def build_demo_index(
    *,
    corpus_path: str | Path,
    retriever_mode: DemoRetrieverMode,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    dense_threshold: float = DEFAULT_DENSE_THRESHOLD,
    rrf_constant: int = DEFAULT_RRF_CONSTANT,
    embedding_timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[ScholarshipSearchIndex, int]:
    """Load a scholarship corpus and construct a retriever."""
    path = Path(corpus_path)
    records = load_scholarships(path)

    if retriever_mode == DemoRetrieverMode.BM25_ONLY:
        index: ScholarshipSearchIndex = (
            BM25ScholarshipIndex(records)
        )
    else:
        embedder = OllamaEmbeddingClient(
            model=embedding_model,
            timeout=embedding_timeout,
        )

        index = HybridScholarshipIndex(
            records,
            embedder,
            rrf_constant=rrf_constant,
            minimum_dense_score=dense_threshold,
        )

    return index, len(records)


def run_demo_workflow(
    *,
    query: str,
    profile: StudentProfile,
    index: ScholarshipSearchIndex,
    corpus_path: str | Path,
    corpus_size: int,
    execution_mode: DemoExecutionMode,
    retriever_mode: DemoRetrieverMode,
    as_of: date,
    generator_model: str = DEFAULT_GENERATOR_MODEL,
    generator_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    top_k: int = DEFAULT_TOP_K,
) -> DemoExecution:
    """Run the bounded ScholarAgent workflow for the demo."""
    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError(
            "The scholarship query must not be empty."
        )

    if execution_mode == DemoExecutionMode.FAST_VERIFIED:

        def demo_generator(_: str) -> str:
            return (
                "Fast demonstration mode intentionally "
                "returns an uncited draft so the existing "
                "verification and fallback pathway is shown."
            )

        generator = demo_generator
        generator_name = (
            "fast-demo-deterministic-fallback"
        )
        external_llm_calls = 0

    else:

        def ollama_generator(prompt: str) -> str:
            return generate(
                prompt,
                model=generator_model,
                temperature=0.0,
                timeout=generator_timeout,
            )

        generator = ollama_generator
        generator_name = generator_model
        external_llm_calls = -1

    result = run_agentic_rag(
        query=normalized_query,
        profile=profile,
        index=index,
        generator=generator,
        generator_name=generator_name,
        as_of=as_of,
        top_k=top_k,
        max_retrieval_attempts=2,
        max_generation_attempts=2,
    )

    if external_llm_calls < 0:
        external_llm_calls = result.generation_calls

    return DemoExecution(
        execution_mode=execution_mode,
        retriever_mode=retriever_mode,
        corpus_path=str(Path(corpus_path)),
        corpus_size=corpus_size,
        external_llm_calls=external_llm_calls,
        result=result,
    )
