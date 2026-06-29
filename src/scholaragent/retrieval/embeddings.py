"""Embedding providers used by ScholarAgent retrieval components."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingClientError(RuntimeError):
    """Raised when an embedding request or response is invalid."""


class EmbeddingProvider(Protocol):
    """Interface implemented by ScholarAgent embedding providers."""

    def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Return one embedding vector for every supplied text."""


class OllamaEmbeddingClient:
    """Minimal client for Ollama's local embedding endpoint."""

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        if not model.strip():
            raise ValueError("Embedding model name must not be empty.")

        if timeout <= 0:
            raise ValueError("Embedding timeout must be positive.")

        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Embed a non-empty sequence of non-empty strings."""
        normalized_texts = [
            text.strip()
            for text in texts
        ]

        if not normalized_texts:
            raise ValueError(
                "At least one text is required for embedding."
            )

        if any(not text for text in normalized_texts):
            raise ValueError(
                "Embedding inputs must not contain empty text."
            )

        payload = {
            "model": self.model,
            "input": normalized_texts,
        }

        request = Request(
            f"{self.base_url}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                raw_response = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode(
                "utf-8",
                errors="replace",
            )
            raise EmbeddingClientError(
                f"Ollama embedding request failed with "
                f"HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise EmbeddingClientError(
                f"Could not reach Ollama embedding API: {exc}"
            ) from exc

        try:
            result = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise EmbeddingClientError(
                "Ollama returned invalid JSON."
            ) from exc

        vectors = result.get("embeddings")

        if not isinstance(vectors, list):
            raise EmbeddingClientError(
                "Ollama response does not contain embeddings."
            )

        if len(vectors) != len(normalized_texts):
            raise EmbeddingClientError(
                "Ollama returned an unexpected number of embeddings."
            )

        validated: list[list[float]] = []
        expected_dimension: int | None = None

        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise EmbeddingClientError(
                    "Ollama returned an invalid embedding vector."
                )

            if not all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                for value in vector
            ):
                raise EmbeddingClientError(
                    "Embedding vector contains invalid values."
                )

            numeric_vector = [
                float(value)
                for value in vector
            ]

            if expected_dimension is None:
                expected_dimension = len(numeric_vector)
            elif len(numeric_vector) != expected_dimension:
                raise EmbeddingClientError(
                    "Embedding dimensions are inconsistent."
                )

            validated.append(numeric_vector)

        return validated
