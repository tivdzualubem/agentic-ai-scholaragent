"""Conventional single-pass RAG baseline for ScholarAgent."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scholaragent.grounding import (
    GroundedScholarshipReport,
    build_grounded_report,
)
from scholaragent.pipeline import search_and_screen
from scholaragent.retrieval import ScholarshipSearchIndex
from scholaragent.schemas import StudentProfile


TextGenerator = Callable[[str], str]

CITATION_PATTERN = re.compile(
    r"\[([a-z0-9][a-z0-9-]*:[a-z_]+)\]"
)


class RAGCitationAudit(BaseModel):
    """Citation audit for a generated RAG answer."""

    model_config = ConfigDict(extra="forbid")

    citation_required: bool
    passed: bool
    cited_ids: list[str]
    available_ids: list[str]
    invalid_ids: list[str]
    errors: list[str]


class SinglePassRAGResult(BaseModel):
    """Structured output from the conventional RAG baseline."""

    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "completed",
        "abstained",
        "citation_failed",
    ]
    query: str
    generator_name: str
    answer: str

    retrieval_calls: int = Field(ge=0)
    generation_calls: int = Field(ge=0)
    query_rewrites: int = Field(ge=0)

    retrieved_count: int = Field(ge=0)
    grounded_report: GroundedScholarshipReport
    citation_audit: RAGCitationAudit


def build_single_pass_prompt(
    *,
    query: str,
    profile: StudentProfile,
    grounded_report: GroundedScholarshipReport,
) -> str:
    """Construct a citation-constrained single-pass RAG prompt."""
    context_lines: list[str] = []

    for candidate_number, candidate in enumerate(
        grounded_report.candidates,
        start=1,
    ):
        context_lines.append(
            f"CANDIDATE {candidate_number}: {candidate.title}"
        )
        context_lines.append(
            f"Official source: {candidate.official_url}"
        )
        context_lines.append(
            "Eligibility screening: "
            f"{candidate.eligibility_status}"
        )

        for claim in candidate.claims:
            markers = " ".join(
                f"[{citation_id}]"
                for citation_id in claim.citation_ids
            )
            context_lines.append(
                f"- {claim.text} {markers}"
            )

        context_lines.append("")

    profile_json = json.dumps(
        profile.model_dump(mode="json"),
        indent=2,
        ensure_ascii=False,
    )

    evidence_context = "\n".join(context_lines).strip()

    return f"""You are the answer generator in a conventional single-pass scholarship RAG system.

Use only the VERIFIED EVIDENCE below. Do not use outside knowledge.
Do not invent eligibility rules, benefits, deadlines, or universities.

Answer the student's question in 2 to 4 concise bullet points.

Citation rules:
1. Every factual bullet must end with at least one exact citation marker.
2. Copy citation markers exactly as provided.
3. Do not create new citation identifiers.
4. Clearly state that final eligibility must be confirmed on the official source.
5. Do not include a bibliography or references section.

STUDENT QUESTION:
{query}

STUDENT PROFILE:
{profile_json}

VERIFIED EVIDENCE:
{evidence_context}

ANSWER:
"""


def audit_generated_answer(
    *,
    answer: str,
    grounded_report: GroundedScholarshipReport,
) -> RAGCitationAudit:
    """Check whether generated citation identifiers are valid."""
    available_ids = sorted(
        {
            evidence.citation_id
            for candidate in grounded_report.candidates
            for evidence in candidate.evidence
        }
    )

    cited_ids = list(
        dict.fromkeys(
            CITATION_PATTERN.findall(answer)
        )
    )

    invalid_ids = sorted(
        set(cited_ids) - set(available_ids)
    )

    errors: list[str] = []

    if not cited_ids:
        errors.append(
            "The generated answer contains no citation markers."
        )

    if invalid_ids:
        errors.append(
            "The generated answer contains unknown citation IDs: "
            + ", ".join(invalid_ids)
        )

    return RAGCitationAudit(
        citation_required=True,
        passed=not errors,
        cited_ids=cited_ids,
        available_ids=available_ids,
        invalid_ids=invalid_ids,
        errors=errors,
    )


def run_single_pass_rag(
    *,
    query: str,
    profile: StudentProfile,
    index: ScholarshipSearchIndex,
    generator: TextGenerator,
    generator_name: str,
    as_of: date,
    top_k: int = 3,
) -> SinglePassRAGResult:
    """Run one retrieval pass and one generation pass."""
    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError("RAG query must not be empty.")

    if not generator_name.strip():
        raise ValueError(
            "Generator name must not be empty."
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    search_report = search_and_screen(
        query=normalized_query,
        profile=profile,
        index=index,
        k=top_k,
        as_of=as_of,
    )

    grounded_report = build_grounded_report(
        search_report,
        as_of=as_of,
        include_ineligible=False,
    )

    if not grounded_report.candidates:
        return SinglePassRAGResult(
            status="abstained",
            query=normalized_query,
            generator_name=generator_name,
            answer=(
                "No sufficiently supported scholarship "
                "candidate was retrieved."
            ),
            retrieval_calls=1,
            generation_calls=0,
            query_rewrites=0,
            retrieved_count=search_report.retrieved_count,
            grounded_report=grounded_report,
            citation_audit=RAGCitationAudit(
                citation_required=False,
                passed=True,
                cited_ids=[],
                available_ids=[],
                invalid_ids=[],
                errors=[],
            ),
        )

    if not grounded_report.all_citations_verified:
        raise RuntimeError(
            "Grounded evidence failed verification before "
            "LLM generation."
        )

    prompt = build_single_pass_prompt(
        query=normalized_query,
        profile=profile,
        grounded_report=grounded_report,
    )

    generated_answer = generator(prompt)

    if not isinstance(generated_answer, str):
        raise TypeError(
            "The text generator must return a string."
        )

    generated_answer = generated_answer.strip()

    if not generated_answer:
        raise ValueError(
            "The text generator returned an empty answer."
        )

    citation_audit = audit_generated_answer(
        answer=generated_answer,
        grounded_report=grounded_report,
    )

    status: Literal[
        "completed",
        "citation_failed",
    ] = (
        "completed"
        if citation_audit.passed
        else "citation_failed"
    )

    return SinglePassRAGResult(
        status=status,
        query=normalized_query,
        generator_name=generator_name,
        answer=generated_answer,
        retrieval_calls=1,
        generation_calls=1,
        query_rewrites=0,
        retrieved_count=search_report.retrieved_count,
        grounded_report=grounded_report,
        citation_audit=citation_audit,
    )
