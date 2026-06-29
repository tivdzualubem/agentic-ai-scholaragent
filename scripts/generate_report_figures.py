"""Generate report-ready ScholarAgent evaluation figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


TABLES = Path("reports/tables")
OUTPUT = Path("reports/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_csv(filename: str) -> list[dict[str, str]]:
    path = TABLES / filename

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    png_path = OUTPUT / f"{filename}.png"
    pdf_path = OUTPUT / f"{filename}.pdf"

    figure.tight_layout()
    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


def grouped_bar(
    categories: list[str],
    series: list[tuple[str, list[float]]],
    title: str,
    ylabel: str,
    filename: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    count = len(series)
    width = 0.8 / count
    positions = list(range(len(categories)))

    for index, (label, values) in enumerate(series):
        offset = (
            index - (count - 1) / 2
        ) * width

        bars = axis.bar(
            [
                position + offset
                for position in positions
            ],
            values,
            width,
            label=label,
        )

        axis.bar_label(
            bars,
            fmt="%.2f",
            padding=3,
            fontsize=8,
        )

    axis.set_xticks(
        positions,
        categories,
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.legend()

    if ylim is not None:
        axis.set_ylim(*ylim)

    save_figure(figure, filename)


retrieval = read_csv("retrieval_k3.csv")

retriever_names = [
    row["retriever"]
    .replace("_", " ")
    .upper()
    for row in retrieval
]

grouped_bar(
    retriever_names,
    [
        (
            "Precision@3",
            [
                float(row["precision_at_k"])
                for row in retrieval
            ],
        ),
        (
            "Recall@3",
            [
                float(row["recall_at_k"])
                for row in retrieval
            ],
        ),
        (
            "MRR",
            [
                float(row["mrr"])
                for row in retrieval
            ],
        ),
        (
            "Top-1 hit rate",
            [
                float(row["top1_hit_rate"])
                for row in retrieval
            ],
        ),
    ],
    "Held-Out Retrieval Performance at k=3",
    "Score",
    "retrieval_k3_comparison",
    (0.0, 1.15),
)

eligibility = read_csv(
    "eligibility_by_status.csv"
)

eligibility_labels = [
    row["status"].replace("_", " ").title()
    for row in eligibility
]

figure, axis = plt.subplots(
    figsize=(9, 5.5)
)

bars = axis.bar(
    eligibility_labels,
    [
        float(row["f1"])
        for row in eligibility
    ],
)

axis.bar_label(
    bars,
    fmt="%.2f",
    padding=3,
)

axis.set_title(
    "Held-Out Eligibility F1 by Status"
)
axis.set_ylabel("F1 score")
axis.set_ylim(0.0, 1.15)
axis.tick_params(
    axis="x",
    rotation=15,
)

save_figure(
    figure,
    "eligibility_f1_by_status",
)

rag = read_csv("rag_comparison.csv")

rag_names = [
    row["system"]
    .replace("_", " ")
    .title()
    for row in rag
]

grouped_bar(
    rag_names,
    [
        (
            "Completion",
            [
                float(
                    row[
                        "positive_completion_rate"
                    ]
                )
                for row in rag
            ],
        ),
        (
            "Citation pass",
            [
                float(
                    row[
                        "positive_citation_pass_rate"
                    ]
                )
                for row in rag
            ],
        ),
        (
            "Relevant grounding",
            [
                float(
                    row[
                        "positive_relevant_grounding_rate"
                    ]
                )
                for row in rag
            ],
        ),
        (
            "Relevant citation",
            [
                float(
                    row[
                        "positive_relevant_citation_rate"
                    ]
                )
                for row in rag
            ],
        ),
        (
            "No-result accuracy",
            [
                float(row["no_result_accuracy"])
                for row in rag
            ],
        ),
    ],
    "Single-Pass versus Agentic RAG",
    "Rate",
    "rag_quality_comparison",
    (0.0, 1.15),
)

ablation = read_csv(
    "verification_ablation.csv"
)

ablation_names = [
    row["configuration"]
    .replace("_", " ")
    .title()
    for row in ablation
]

grouped_bar(
    ablation_names,
    [
        (
            "Citation pass",
            [
                float(row["citation_pass_rate"])
                for row in ablation
            ],
        ),
        (
            "Relevant citation",
            [
                float(
                    row[
                        "relevant_citation_rate"
                    ]
                )
                for row in ablation
            ],
        ),
        (
            "Unsafe acceptance",
            [
                float(
                    row[
                        "unsafe_acceptance_rate"
                    ]
                )
                for row in ablation
            ],
        ),
        (
            "Fallback rate",
            [
                float(row["fallback_rate"])
                for row in ablation
            ],
        ),
    ],
    "Verification-Stage Ablation",
    "Rate",
    "verification_ablation",
    (0.0, 1.15),
)

runtime = read_csv("runtime_and_cost.csv")

runtime_names = [
    row["system"]
    .replace("_", " ")
    .title()
    for row in runtime
]

figure, axis = plt.subplots(
    figsize=(8, 5.5)
)

bars = axis.bar(
    runtime_names,
    [
        float(row["mean_latency_seconds"])
        for row in runtime
    ],
)

axis.bar_label(
    bars,
    fmt="%.1f s",
    padding=3,
)

axis.set_title(
    "Mean End-to-End Latency"
)
axis.set_ylabel("Seconds per benchmark case")

save_figure(
    figure,
    "mean_latency_comparison",
)

figure, axis = plt.subplots(
    figsize=(8, 5.5)
)

bars = axis.bar(
    runtime_names,
    [
        int(row["total_generation_calls"])
        for row in runtime
    ],
)

axis.bar_label(
    bars,
    fmt="%d",
    padding=3,
)

axis.set_title(
    "Total Local Generation Calls"
)
axis.set_ylabel(
    "Generation calls across 24 cases"
)

save_figure(
    figure,
    "generation_call_comparison",
)

print()
print(
    "Figure pairs generated:",
    len(list(OUTPUT.glob("*.png"))),
)
