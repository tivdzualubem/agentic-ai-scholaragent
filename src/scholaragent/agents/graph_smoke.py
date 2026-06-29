"""Minimal LangGraph integration used to verify local orchestration."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class SmokeState(TypedDict):
    """State passed between the demonstration graph nodes."""

    message: str
    completed: bool


def mark_completed(state: SmokeState) -> SmokeState:
    """Mark the graph execution as complete."""
    return {
        "message": f"{state['message']} -> LangGraph working",
        "completed": True,
    }


def build_smoke_graph():
    """Compile the smallest valid ScholarAgent state graph."""
    builder = StateGraph(SmokeState)

    builder.add_node("mark_completed", mark_completed)
    builder.add_edge(START, "mark_completed")
    builder.add_edge("mark_completed", END)

    return builder.compile()
