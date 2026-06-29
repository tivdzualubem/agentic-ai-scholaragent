"""Bounded LangGraph workflow for scholarship retrieval and screening."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from scholaragent.eligibility import EligibilityStatus
from scholaragent.grounding import (
    GroundedScholarshipReport,
    build_grounded_report,
)
from scholaragent.pipeline import search_and_screen
from scholaragent.retrieval import ScholarshipSearchIndex
from scholaragent.retrieval.bm25 import tokenize
from scholaragent.schemas import StudentProfile


class AgentCandidate(BaseModel):
    """Compact candidate returned by the agent workflow."""

    model_config = ConfigDict(extra="forbid")

    scholarship_id: str
    title: str
    retrieval_rank: int = Field(ge=1)
    retrieval_score: float
    eligibility_status: str


class ScholarAgentOutcome(BaseModel):
    """Final structured result of one ScholarAgent execution."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "abstained"]
    original_query: str
    final_query: str
    attempts: int = Field(ge=1)
    rewrites: list[str]
    evidence_sufficient: bool
    explanation: str
    candidates: list[AgentCandidate]
    grounded_report: GroundedScholarshipReport | None = None


class ScholarAgentState(TypedDict, total=False):
    """Shared state passed through the LangGraph nodes."""

    original_query: str
    working_query: str
    profile: StudentProfile
    as_of: date
    top_k: int
    max_attempts: int
    attempts: int
    rewrites: list[str]
    report: Any
    evidence_sufficient: bool
    grade_reason: str
    outcome: ScholarAgentOutcome


def _enum_value(value: Any) -> str:
    """Return a stable string representation for enum-like values."""
    return str(getattr(value, "value", value))


def _build_profile_query(profile: StudentProfile) -> str:
    """Create deterministic retrieval constraints from a student profile."""
    parts: list[str] = []

    degree = _enum_value(profile.target_degree_level)
    if degree:
        parts.append(degree)

    parts.extend(profile.fields_of_study[:3])
    parts.extend(profile.preferred_countries[:2])

    if profile.requires_full_funding:
        parts.append("fully funded")

    return " ".join(
        part.strip()
        for part in parts
        if isinstance(part, str) and part.strip()
    )


def build_scholar_agent_graph(
    index: ScholarshipSearchIndex,
    *,
    default_top_k: int = 5,
):
    """Compile the bounded scholarship-search LangGraph."""

    def plan_query(
        state: ScholarAgentState,
    ) -> ScholarAgentState:
        query = state["original_query"].strip()

        if not query:
            raise ValueError("ScholarAgent query must not be empty.")

        max_attempts = state.get("max_attempts", 2)
        top_k = state.get("top_k", default_top_k)

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        return {
            "working_query": query,
            "attempts": 0,
            "rewrites": [],
            "max_attempts": max_attempts,
            "top_k": top_k,
        }

    def retrieve_and_screen(
        state: ScholarAgentState,
    ) -> ScholarAgentState:
        attempts = state["attempts"] + 1

        report = search_and_screen(
            query=state["working_query"],
            profile=state["profile"],
            index=index,
            k=state["top_k"],
            as_of=state["as_of"],
        )

        return {
            "attempts": attempts,
            "report": report,
        }

    def grade_evidence(
        state: ScholarAgentState,
    ) -> ScholarAgentState:
        results = state["report"].results

        viable_results = [
            result
            for result in results
            if result.assessment.status
            is not EligibilityStatus.NOT_ELIGIBLE
        ]

        if viable_results:
            return {
                "evidence_sufficient": True,
                "grade_reason": (
                    "At least one retrieved scholarship is viable "
                    "for the supplied profile."
                ),
            }

        if not results:
            return {
                "evidence_sufficient": False,
                "grade_reason": (
                    "No scholarship evidence matched the current query."
                ),
            }

        top_retrieved = min(
            results,
            key=lambda item: item.retrieval_rank,
        )

        if (
            top_retrieved.assessment.status
            is EligibilityStatus.NOT_ELIGIBLE
        ):
            return {
                "evidence_sufficient": True,
                "grade_reason": (
                    "The top-ranked retrieved scholarship is "
                    "relevant but contains hard eligibility "
                    "failures. Preserve it as explanatory evidence "
                    "without recommending it."
                ),
            }

        return {
            "evidence_sufficient": False,
            "grade_reason": (
                "Retrieved evidence was insufficient for a "
                "recommendation or a grounded eligibility explanation."
            ),
        }

    def route_after_grading(
        state: ScholarAgentState,
    ) -> Literal["rewrite_query", "finalize"]:
        if state["evidence_sufficient"]:
            return "finalize"

        if state["attempts"] >= state["max_attempts"]:
            return "finalize"

        # An explicit out-of-domain query must not be rewritten using
        # profile terms because that could manufacture false relevance.
        if (
            not state["report"].results
            and tokenize(state["working_query"])
        ):
            return "finalize"

        return "rewrite_query"

    def rewrite_query(
        state: ScholarAgentState,
    ) -> ScholarAgentState:
        profile_query = _build_profile_query(state["profile"])

        rewritten = " ".join(
            part
            for part in [
                state["original_query"].strip(),
                profile_query,
            ]
            if part
        )

        if rewritten == state["working_query"]:
            return {
                "working_query": rewritten,
                "rewrites": state["rewrites"],
            }

        return {
            "working_query": rewritten,
            "rewrites": [
                *state["rewrites"],
                rewritten,
            ],
        }

    def finalize(
        state: ScholarAgentState,
    ) -> ScholarAgentState:
        results = state["report"].results

        candidates = [
            AgentCandidate(
                scholarship_id=(
                    result.scholarship.scholarship_id
                ),
                title=result.scholarship.title,
                retrieval_rank=result.retrieval_rank,
                retrieval_score=result.retrieval_score,
                eligibility_status=(
                    result.assessment.status.value
                ),
            )
            for result in results
            if result.assessment.status
            is not EligibilityStatus.NOT_ELIGIBLE
        ]

        grounded_report = build_grounded_report(
            state["report"],
            as_of=state["as_of"],
            include_explanatory_ineligible=True,
        )

        completed = (
            state["evidence_sufficient"]
            and bool(grounded_report.candidates)
        )

        if completed and candidates:
            explanation = (
                f"Found {len(candidates)} viable scholarship "
                "candidate(s) supported by retrieved evidence."
            )
            status: Literal["completed", "abstained"] = (
                "completed"
            )
        elif completed:
            explanation = (
                "Found a highly relevant scholarship match, but "
                "it contains hard eligibility failures. It is "
                "retained only as explanatory evidence and is not "
                "an actionable recommendation."
            )
            status = "completed"
        else:
            explanation = (
                "ScholarAgent abstained because it could not find "
                "sufficient evidence for either a viable scholarship "
                "or a relevant eligibility explanation. "
                f"Reason: {state['grade_reason']}"
            )
            status = "abstained"

        outcome = ScholarAgentOutcome(
            status=status,
            original_query=state["original_query"],
            final_query=state["working_query"],
            attempts=state["attempts"],
            rewrites=state["rewrites"],
            evidence_sufficient=(
                state["evidence_sufficient"]
            ),
            explanation=explanation,
            candidates=candidates,
            grounded_report=grounded_report,
        )

        return {"outcome": outcome}

    builder = StateGraph(ScholarAgentState)

    builder.add_node("plan_query", plan_query)
    builder.add_node(
        "retrieve_and_screen",
        retrieve_and_screen,
    )
    builder.add_node("grade_evidence", grade_evidence)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "plan_query")
    builder.add_edge(
        "plan_query",
        "retrieve_and_screen",
    )
    builder.add_edge(
        "retrieve_and_screen",
        "grade_evidence",
    )
    builder.add_conditional_edges(
        "grade_evidence",
        route_after_grading,
        {
            "rewrite_query": "rewrite_query",
            "finalize": "finalize",
        },
    )
    builder.add_edge(
        "rewrite_query",
        "retrieve_and_screen",
    )
    builder.add_edge("finalize", END)

    return builder.compile()
