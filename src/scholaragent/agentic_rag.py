"""Agentic RAG workflow with retrieval rewriting and citation repair."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from scholaragent.agents.scholar_graph import (
    ScholarAgentOutcome,
    build_scholar_agent_graph,
)
from scholaragent.grounding import GroundedScholarshipReport
from scholaragent.rag_baseline import (
    RAGCitationAudit,
    audit_generated_answer,
    build_single_pass_prompt,
)
from scholaragent.retrieval import ScholarshipSearchIndex
from scholaragent.schemas import StudentProfile


TextGenerator = Callable[[str], str]


class AgenticRAGResult(BaseModel):
    """Final result from the Agentic RAG workflow."""

    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "completed",
        "completed_fallback",
        "abstained",
        "citation_failed",
    ]

    query: str
    final_query: str
    generator_name: str
    answer: str

    retrieval_calls: int = Field(ge=0)
    generation_calls: int = Field(ge=0)
    query_rewrites: int = Field(ge=0)
    repair_attempts: int = Field(ge=0)
    fallback_used: bool = False

    grounded_report: GroundedScholarshipReport
    citation_audit: RAGCitationAudit
    audit_history: list[RAGCitationAudit]


class AgenticRAGState(TypedDict, total=False):
    """State shared by the Agentic RAG LangGraph nodes."""

    query: str
    profile: StudentProfile
    as_of: date

    top_k: int
    max_retrieval_attempts: int
    max_generation_attempts: int

    retrieval_outcome: ScholarAgentOutcome
    grounded_report: GroundedScholarshipReport

    answer: str
    generation_attempts: int
    audit: RAGCitationAudit
    audit_history: list[RAGCitationAudit]

    result: AgenticRAGResult


def build_repair_prompt(
    *,
    query: str,
    profile: StudentProfile,
    grounded_report: GroundedScholarshipReport,
    previous_answer: str,
    audit: RAGCitationAudit,
) -> str:
    """Create a strict correction prompt after citation failure."""
    base_prompt = build_single_pass_prompt(
        query=query,
        profile=profile,
        grounded_report=grounded_report,
    )

    allowed_ids = "\n".join(
        f"- [{citation_id}]"
        for citation_id in audit.available_ids
    )

    audit_errors = "\n".join(
        f"- {error}"
        for error in audit.errors
    )

    return f"""{base_prompt}

The previous answer failed citation verification.

PREVIOUS ANSWER:
{previous_answer}

VERIFICATION ERRORS:
{audit_errors}

ALLOWED CITATION MARKERS:
{allowed_ids}

Rewrite the answer completely.

Mandatory repair rules:
1. Produce only 2 to 4 concise bullet points.
2. Every factual bullet must end with one or more allowed markers.
3. Copy the allowed markers exactly, including brackets.
4. Do not repeat the question, profile, evidence, or instructions.
5. Do not create headings, a bibliography, or new identifiers.
6. State that final eligibility must be confirmed on the official source.

REPAIRED ANSWER:
"""



def build_deterministic_fallback_answer(
    grounded_report: GroundedScholarshipReport,
) -> str:
    """Build a citation-safe extractive answer from verified claims."""
    if not grounded_report.candidates:
        raise ValueError(
            "A fallback answer requires at least one "
            "grounded candidate."
        )

    candidate = grounded_report.candidates[0]

    preferred_suffixes = (
        ":claim:identity",
        ":claim:location",
        ":claim:funding",
        ":claim:eligibility",
    )

    selected_claims = []

    for suffix in preferred_suffixes:
        claim = next(
            (
                item
                for item in candidate.claims
                if item.claim_id.endswith(suffix)
            ),
            None,
        )

        if claim is not None:
            selected_claims.append(claim)

    if len(selected_claims) < 2:
        selected_claims = candidate.claims[:4]

    bullets: list[str] = []

    for claim in selected_claims[:4]:
        claim_text = claim.text

        if claim.claim_id.endswith(
            ":claim:eligibility"
        ):
            claim_text += (
                " Confirm final eligibility on the "
                "official source before applying."
            )

        markers = " ".join(
            f"[{citation_id}]"
            for citation_id in claim.citation_ids
        )

        bullets.append(
            f"- {claim_text} {markers}"
        )

    if not 2 <= len(bullets) <= 4:
        raise RuntimeError(
            "Verified fallback could not construct "
            "2 to 4 cited bullets."
        )

    return "\n".join(bullets)


def build_agentic_rag_graph(
    *,
    index: ScholarshipSearchIndex,
    generator: TextGenerator,
    generator_name: str,
):
    """Compile the complete retrieval and citation-repair graph."""
    if not generator_name.strip():
        raise ValueError(
            "Generator name must not be empty."
        )

    retrieval_graph = build_scholar_agent_graph(index)

    def validate_request(
        state: AgenticRAGState,
    ) -> AgenticRAGState:
        query = state["query"].strip()

        if not query:
            raise ValueError(
                "Agentic RAG query must not be empty."
            )

        top_k = state.get("top_k", 3)
        max_retrieval_attempts = state.get(
            "max_retrieval_attempts",
            2,
        )
        max_generation_attempts = state.get(
            "max_generation_attempts",
            2,
        )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        if max_retrieval_attempts < 1:
            raise ValueError(
                "max_retrieval_attempts must be at least 1."
            )

        if max_generation_attempts < 1:
            raise ValueError(
                "max_generation_attempts must be at least 1."
            )

        return {
            "query": query,
            "top_k": top_k,
            "max_retrieval_attempts": (
                max_retrieval_attempts
            ),
            "max_generation_attempts": (
                max_generation_attempts
            ),
            "generation_attempts": 0,
            "audit_history": [],
        }

    def run_retrieval_agent(
        state: AgenticRAGState,
    ) -> AgenticRAGState:
        retrieval_state = retrieval_graph.invoke(
            {
                "original_query": state["query"],
                "profile": state["profile"],
                "as_of": state["as_of"],
                "top_k": state["top_k"],
                "max_attempts": (
                    state["max_retrieval_attempts"]
                ),
            }
        )

        outcome = ScholarAgentOutcome.model_validate(
            retrieval_state["outcome"]
        )

        if outcome.grounded_report is None:
            raise RuntimeError(
                "Retrieval agent produced no grounded report."
            )

        return {
            "retrieval_outcome": outcome,
            "grounded_report": outcome.grounded_report,
        }

    def route_after_retrieval(
        state: AgenticRAGState,
    ) -> Literal[
        "generate_answer",
        "finalize_abstention",
    ]:
        outcome = state["retrieval_outcome"]

        if (
            outcome.status == "completed"
            and state["grounded_report"].candidates
        ):
            return "generate_answer"

        return "finalize_abstention"

    def generate_answer(
        state: AgenticRAGState,
    ) -> AgenticRAGState:
        attempts = state["generation_attempts"]

        if attempts == 0:
            prompt = build_single_pass_prompt(
                query=state["query"],
                profile=state["profile"],
                grounded_report=state[
                    "grounded_report"
                ],
            )
        else:
            prompt = build_repair_prompt(
                query=state["query"],
                profile=state["profile"],
                grounded_report=state[
                    "grounded_report"
                ],
                previous_answer=state["answer"],
                audit=state["audit"],
            )

        answer = generator(prompt)

        if not isinstance(answer, str):
            raise TypeError(
                "The text generator must return a string."
            )

        answer = answer.strip()

        if not answer:
            raise ValueError(
                "The text generator returned an empty answer."
            )

        return {
            "answer": answer,
            "generation_attempts": attempts + 1,
        }

    def audit_answer(
        state: AgenticRAGState,
    ) -> AgenticRAGState:
        audit = audit_generated_answer(
            answer=state["answer"],
            grounded_report=state[
                "grounded_report"
            ],
        )

        return {
            "audit": audit,
            "audit_history": [
                *state["audit_history"],
                audit,
            ],
        }

    def route_after_audit(
        state: AgenticRAGState,
    ) -> Literal[
        "generate_answer",
        "finalize_completed",
        "finalize_failed",
    ]:
        if state["audit"].passed:
            return "finalize_completed"

        if (
            state["generation_attempts"]
            < state["max_generation_attempts"]
        ):
            return "generate_answer"

        return "finalize_failed"

    def make_result(
        state: AgenticRAGState,
        *,
        status: Literal[
            "completed",
            "citation_failed",
        ],
    ) -> AgenticRAGResult:
        retrieval = state["retrieval_outcome"]

        return AgenticRAGResult(
            status=status,
            query=state["query"],
            final_query=retrieval.final_query,
            generator_name=generator_name,
            answer=state["answer"],
            retrieval_calls=retrieval.attempts,
            generation_calls=state[
                "generation_attempts"
            ],
            query_rewrites=len(retrieval.rewrites),
            repair_attempts=max(
                state["generation_attempts"] - 1,
                0,
            ),
            grounded_report=state[
                "grounded_report"
            ],
            citation_audit=state["audit"],
            audit_history=state["audit_history"],
        )

    def finalize_completed(
        state: AgenticRAGState,
    ) -> AgenticRAGState:
        return {
            "result": make_result(
                state,
                status="completed",
            )
        }

    def finalize_failed(
        state: AgenticRAGState,
    ) -> AgenticRAGState:
        return {
            "result": make_result(
                state,
                status="citation_failed",
            )
        }

    def finalize_abstention(
        state: AgenticRAGState,
    ) -> AgenticRAGState:
        retrieval = state["retrieval_outcome"]

        audit = RAGCitationAudit(
            citation_required=False,
            passed=True,
            cited_ids=[],
            available_ids=[],
            invalid_ids=[],
            errors=[],
        )

        result = AgenticRAGResult(
            status="abstained",
            query=state["query"],
            final_query=retrieval.final_query,
            generator_name=generator_name,
            answer=(
                "ScholarAgent abstained because it could "
                "not retrieve sufficient verified evidence."
            ),
            retrieval_calls=retrieval.attempts,
            generation_calls=0,
            query_rewrites=len(retrieval.rewrites),
            repair_attempts=0,
            grounded_report=state[
                "grounded_report"
            ],
            citation_audit=audit,
            audit_history=[],
        )

        return {"result": result}

    builder = StateGraph(AgenticRAGState)

    builder.add_node(
        "validate_request",
        validate_request,
    )
    builder.add_node(
        "run_retrieval_agent",
        run_retrieval_agent,
    )
    builder.add_node(
        "generate_answer",
        generate_answer,
    )
    builder.add_node(
        "audit_answer",
        audit_answer,
    )
    builder.add_node(
        "finalize_completed",
        finalize_completed,
    )
    builder.add_node(
        "finalize_failed",
        finalize_failed,
    )
    builder.add_node(
        "finalize_abstention",
        finalize_abstention,
    )

    builder.add_edge(
        START,
        "validate_request",
    )
    builder.add_edge(
        "validate_request",
        "run_retrieval_agent",
    )

    builder.add_conditional_edges(
        "run_retrieval_agent",
        route_after_retrieval,
        {
            "generate_answer": "generate_answer",
            "finalize_abstention": (
                "finalize_abstention"
            ),
        },
    )

    builder.add_edge(
        "generate_answer",
        "audit_answer",
    )

    builder.add_conditional_edges(
        "audit_answer",
        route_after_audit,
        {
            "generate_answer": "generate_answer",
            "finalize_completed": (
                "finalize_completed"
            ),
            "finalize_failed": "finalize_failed",
        },
    )

    builder.add_edge(
        "finalize_completed",
        END,
    )
    builder.add_edge(
        "finalize_failed",
        END,
    )
    builder.add_edge(
        "finalize_abstention",
        END,
    )

    return builder.compile()


def run_agentic_rag(
    *,
    query: str,
    profile: StudentProfile,
    index: ScholarshipSearchIndex,
    generator: TextGenerator,
    generator_name: str,
    as_of: date,
    top_k: int = 3,
    max_retrieval_attempts: int = 2,
    max_generation_attempts: int = 2,
) -> AgenticRAGResult:
    """Run the complete bounded Agentic RAG workflow."""
    graph = build_agentic_rag_graph(
        index=index,
        generator=generator,
        generator_name=generator_name,
    )

    state = graph.invoke(
        {
            "query": query,
            "profile": profile,
            "as_of": as_of,
            "top_k": top_k,
            "max_retrieval_attempts": (
                max_retrieval_attempts
            ),
            "max_generation_attempts": (
                max_generation_attempts
            ),
        }
    )

    result = AgenticRAGResult.model_validate(
        state["result"]
    )

    if result.status != "citation_failed":
        return result

    fallback_answer = build_deterministic_fallback_answer(
        result.grounded_report
    )

    fallback_audit = audit_generated_answer(
        answer=fallback_answer,
        grounded_report=result.grounded_report,
    )

    updated_data = result.model_dump()
    updated_data.update(
        {
            "answer": fallback_answer,
            "fallback_used": True,
            "citation_audit": fallback_audit,
            "audit_history": [
                *result.audit_history,
                fallback_audit,
            ],
        }
    )

    if fallback_audit.passed:
        updated_data["status"] = (
            "completed_fallback"
        )

    return AgenticRAGResult.model_validate(
        updated_data
    )
