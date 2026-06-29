"""Streamlit demonstration application for ScholarAgent."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st
from pydantic import ValidationError

from scholaragent.llm.ollama_client import OllamaClientError
from scholaragent.retrieval import EmbeddingClientError
from scholaragent.ui import (
    DEFAULT_DENSE_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_TOP_K,
    DemoExecution,
    DemoExecutionMode,
    DemoRetrieverMode,
    build_demo_index,
    build_student_profile,
    run_demo_workflow,
)


ROOT = Path(__file__).resolve().parents[3]

CORPUS_PATHS: dict[str, Path] = {
    "Combined demo corpus (15 scholarships)": (
        ROOT / "data" / "demo" / "combined_scholarships.json"
    ),
    "Official development corpus (3 scholarships)": (
        ROOT / "data" / "official" / "official_scholarships.json"
    ),
    "Calibration corpus (6 scholarships)": (
        ROOT / "data" / "calibration" / "calibration_scholarships.json"
    ),
    "Held-out evaluation corpus (6 scholarships)": (
        ROOT / "data" / "held_out" / "held_out_scholarships.json"
    ),
}

CORPUS_EXAMPLES: dict[str, list[str]] = {
    "Combined demo corpus (15 scholarships)": [
        "University of Bristol Think Big Scholarship for an international master's applicant",
        "ETH Excellence Scholarship for an outstanding master's applicant",
    ],
    "Official development corpus (3 scholarships)": [
        "SI Scholarship for Global Professionals for a Nigerian master's applicant",
    ],
    "Calibration corpus (6 scholarships)": [
        "ETH Excellence Scholarship and Opportunity Programme",
    ],
    "Held-out evaluation corpus (6 scholarships)": [
        "University of Bristol Think Big Scholarship 2026",
    ],
}


def readable(value: Any) -> str:
    """Convert enum-like values into display text."""
    raw = getattr(value, "value", value)
    return str(raw).replace("_", " ").title()


@st.cache_resource(show_spinner=False)
def cached_index(
    corpus_path: str,
    retriever_mode: str,
):
    """Cache the scholarship index between page reruns."""
    return build_demo_index(
        corpus_path=corpus_path,
        retriever_mode=DemoRetrieverMode(
            retriever_mode
        ),
    )


def render_candidate(
    candidate: Any,
    position: int,
) -> None:
    """Render one evidence-grounded scholarship."""
    with st.container(border=True):
        st.subheader(
            f"{position}. {candidate.title}"
        )

        first, second, third = st.columns(3)

        first.metric(
            "Eligibility",
            readable(candidate.eligibility_status),
        )
        second.metric(
            "Role",
            readable(candidate.candidate_role),
        )
        third.metric(
            "Evidence verified",
            (
                "Yes"
                if candidate.verification.passed
                else "No"
            ),
        )

        st.write(
            f"**Provider:** {candidate.provider}"
        )
        st.write(
            f"**Assessment:** "
            f"{candidate.assessment_note}"
        )
        st.markdown(
            f"[Official scholarship page]"
            f"({candidate.official_url})"
        )

        with st.expander(
            "Claims and evidence",
            expanded=False,
        ):
            st.markdown("#### Grounded claims")

            for claim in candidate.claims:
                citations = " ".join(
                    f"`[{citation}]`"
                    for citation
                    in claim.citation_ids
                )

                st.markdown(
                    f"- {claim.text} {citations}"
                )

            st.markdown("#### Evidence snippets")

            for evidence in candidate.evidence:
                st.markdown(
                    f"**{evidence.field_name}:** "
                    f"{evidence.text}"
                )
                st.caption(
                    f"Citation ID: "
                    f"{evidence.citation_id}"
                )


def render_execution(
    execution: DemoExecution,
) -> None:
    """Render one completed ScholarAgent execution."""
    result = execution.result

    st.divider()
    st.header("ScholarAgent result")

    st.caption(
        f"Evidence corpus: {Path(execution.corpus_path).name} "
        f"({execution.corpus_size} scholarships)"
    )

    if (
        execution.execution_mode
        == DemoExecutionMode.FAST_VERIFIED
        and result.fallback_used
    ):
        st.info(
            "Fast verified mode intentionally tests the deterministic "
            "verification and fallback path without calling TinyLlama."
        )

    if result.status == "abstained":
        st.info(
            "ScholarAgent abstained because it could not "
            "retrieve sufficient verified evidence."
        )
    elif result.citation_audit.passed:
        st.success(
            "The final response passed deterministic "
            "citation verification."
        )
    else:
        st.error(
            "The final response failed citation verification."
        )

    metrics = st.columns(6)

    metrics[0].metric(
        "Status",
        readable(result.status),
    )
    metrics[1].metric(
        "Candidates",
        len(result.grounded_report.candidates),
    )
    metrics[2].metric(
        "Retrieval calls",
        result.retrieval_calls,
    )
    metrics[3].metric(
        "Generation attempts",
        result.generation_calls,
    )
    metrics[4].metric(
        "External LLM calls",
        execution.external_llm_calls,
    )
    metrics[5].metric(
        "Fallback",
        "Yes" if result.fallback_used else "No",
    )

    st.subheader("Final grounded answer")
    st.markdown(result.answer)

    st.caption(
        f"Citation audit: "
        f"{result.citation_audit.passed} | "
        f"Query rewrites: {result.query_rewrites} | "
        f"Repair attempts: {result.repair_attempts}"
    )

    if result.final_query != result.query:
        st.info(
            "Rewritten retrieval query: "
            f"{result.final_query}"
        )

    candidates = result.grounded_report.candidates

    if candidates:
        st.header("Ranked scholarship evidence")

        for position, candidate in enumerate(
            candidates,
            start=1,
        ):
            render_candidate(
                candidate,
                position,
            )
    else:
        st.warning(
            "No grounded scholarship candidate was found."
        )

    with st.expander(
        "Agent execution trace",
        expanded=False,
    ):
        st.json(
            {
                "execution_mode": (
                    execution.execution_mode.value
                ),
                "retriever_mode": (
                    execution.retriever_mode.value
                ),
                "corpus_size": execution.corpus_size,
                "original_query": result.query,
                "final_query": result.final_query,
                "status": result.status,
                "generator": result.generator_name,
                "retrieval_calls": (
                    result.retrieval_calls
                ),
                "generation_attempts": (
                    result.generation_calls
                ),
                "external_llm_calls": (
                    execution.external_llm_calls
                ),
                "query_rewrites": (
                    result.query_rewrites
                ),
                "repair_attempts": (
                    result.repair_attempts
                ),
                "fallback_used": (
                    result.fallback_used
                ),
                "citation_audit": (
                    result.citation_audit.model_dump(
                        mode="json"
                    )
                ),
                "audit_history": [
                    audit.model_dump(mode="json")
                    for audit in result.audit_history
                ],
            }
        )

    st.caption(
        result.grounded_report.disclaimer
    )


def main() -> None:
    """Render the complete ScholarAgent demo."""
    st.set_page_config(
        page_title="ScholarAgent",
        page_icon="🎓",
        layout="wide",
    )

    st.title("🎓 ScholarAgent")
    st.subheader(
        "Evidence-grounded scholarship discovery "
        "and eligibility verification"
    )

    st.write(
        "ScholarAgent retrieves official scholarship "
        "evidence, screens a structured student profile, "
        "verifies citations and abstains when evidence "
        "is insufficient."
    )

    with st.sidebar:
        st.header("Demo configuration")

        execution_mode = st.selectbox(
            "Execution mode",
            options=list(DemoExecutionMode),
            format_func=lambda item: {
                DemoExecutionMode.FAST_VERIFIED:
                    "Fast verified demo",
                DemoExecutionMode.FULL_AGENTIC:
                    "Full local TinyLlama",
            }[item],
        )

        retriever_mode = st.selectbox(
            "Retriever",
            options=list(DemoRetrieverMode),
            format_func=lambda item: {
                DemoRetrieverMode.HYBRID_RRF:
                    "Hybrid BM25 + dense RRF",
                DemoRetrieverMode.BM25_ONLY:
                    "BM25-only offline mode",
            }[item],
        )

        corpus_label = st.selectbox(
            "Scholarship corpus",
            options=list(CORPUS_PATHS),
            index=0,
        )
        selected_corpus = CORPUS_PATHS[corpus_label]

        st.caption(
            "The combined corpus is for the interactive demo only. "
            "Frozen evaluation splits remain separate."
        )

        with st.expander("Example queries", expanded=False):
            for example in CORPUS_EXAMPLES[corpus_label]:
                st.markdown(f"- {example}")

        st.markdown(
            f"""
**Corpus:** {corpus_label}

**Embedding:** `{DEFAULT_EMBEDDING_MODEL}`

**Generator:** `{DEFAULT_GENERATOR_MODEL}`

**Dense threshold:** `{DEFAULT_DENSE_THRESHOLD}`

**Top-k:** `{DEFAULT_TOP_K}`
"""
        )

        if (
            execution_mode
            == DemoExecutionMode.FULL_AGENTIC
        ):
            st.warning(
                "TinyLlama can take several minutes "
                "on CPU-only hardware."
            )

        if (
            retriever_mode
            == DemoRetrieverMode.HYBRID_RRF
        ):
            st.info(
                "Hybrid retrieval requires the local "
                "Ollama embedding service."
            )

    with st.form(
        "scholaragent_form",
        clear_on_submit=False,
    ):
        st.header("1. Scholarship query")

        query = st.text_area(
            "Information need",
            value=(
                "Fully funded master's scholarship for "
                "a Nigerian AI and data science student"
            ),
            height=90,
        )

        st.header("2. Student profile")

        left, middle, right = st.columns(3)

        with left:
            nationality = st.text_input(
                "Nationality",
                value="Nigerian",
            )
            residence = st.text_input(
                "Country of residence",
                value="Finland",
            )
            degree = st.selectbox(
                "Target degree",
                [
                    "bachelor",
                    "master",
                    "phd",
                    "postdoctoral",
                    "other",
                ],
                index=1,
            )

        with middle:
            fields = st.text_input(
                "Fields of study",
                value=(
                    "Artificial Intelligence, "
                    "Data Science"
                ),
            )
            countries = st.text_input(
                "Preferred countries",
                value="Sweden, Netherlands",
            )
            experience = st.number_input(
                "Years of work experience",
                min_value=0.0,
                max_value=60.0,
                value=2.0,
                step=0.5,
            )

        with right:
            full_funding = st.checkbox(
                "Requires full funding",
                value=True,
            )
            assessment_date = st.date_input(
                "Assessment date",
                value=date(2026, 1, 15),
            )

        gpa_column, language_column = st.columns(2)

        with gpa_column:
            include_gpa = st.checkbox(
                "Include GPA",
                value=True,
            )
            gpa = st.number_input(
                "GPA",
                min_value=0.0,
                value=4.2,
                step=0.1,
            )
            gpa_scale = st.number_input(
                "GPA scale",
                min_value=0.1,
                value=5.0,
                step=0.1,
            )

        with language_column:
            include_language = st.checkbox(
                "Include language score",
                value=True,
            )
            language_test = st.selectbox(
                "Language test",
                [
                    "IELTS",
                    "TOEFL",
                    "PTE",
                    "DUOLINGO",
                ],
            )
            language_score = st.number_input(
                "Language score",
                min_value=0.0,
                value=7.5,
                step=0.5,
            )

        submitted = st.form_submit_button(
            "Run ScholarAgent",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        try:
            profile = build_student_profile(
                nationality=nationality,
                country_of_residence=residence,
                target_degree_level=degree,
                fields_of_study=fields,
                include_gpa=include_gpa,
                gpa=float(gpa),
                gpa_scale=float(gpa_scale),
                include_language_score=(
                    include_language
                ),
                language_test=language_test,
                language_score=float(
                    language_score
                ),
                years_work_experience=float(
                    experience
                ),
                preferred_countries=countries,
                requires_full_funding=full_funding,
            )

            with st.spinner(
                "Loading scholarship evidence..."
            ):
                index, corpus_size = cached_index(
                    str(selected_corpus),
                    retriever_mode.value,
                )

            with st.spinner(
                "Running ScholarAgent..."
            ):
                execution = run_demo_workflow(
                    query=query,
                    profile=profile,
                    index=index,
                    corpus_path=selected_corpus,
                    corpus_size=corpus_size,
                    execution_mode=execution_mode,
                    retriever_mode=retriever_mode,
                    as_of=assessment_date,
                )

            st.session_state[
                "last_execution"
            ] = execution.model_dump(
                mode="json"
            )

        except ValidationError as exc:
            st.error("The student profile is invalid.")
            st.code(str(exc))

        except (
            ValueError,
            EmbeddingClientError,
            OllamaClientError,
        ) as exc:
            st.error(str(exc))

        except Exception as exc:
            st.error(
                "ScholarAgent could not complete the request."
            )
            st.exception(exc)

    stored = st.session_state.get(
        "last_execution"
    )

    if stored is not None:
        stored_execution = DemoExecution.model_validate(stored)

        configuration_matches = (
            stored_execution.corpus_path
            == str(selected_corpus)
            and stored_execution.execution_mode
            == execution_mode
            and stored_execution.retriever_mode
            == retriever_mode
        )

        if configuration_matches:
            render_execution(stored_execution)
        else:
            st.info(
                "The configuration changed. Run ScholarAgent again "
                "to generate a result using the selected settings."
            )


if __name__ == "__main__":
    main()
