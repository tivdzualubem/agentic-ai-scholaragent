"""Tests for the minimal Ollama API client."""

from __future__ import annotations

import json
from unittest.mock import patch
from urllib.error import URLError

import pytest

from scholaragent.llm.ollama_client import (
    OllamaClientError,
    generate,
    list_models,
)


class FakeResponse:
    """Small context-manager replacement for urllib responses."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_list_models_returns_model_names() -> None:
    """Installed model names are extracted from the Ollama response."""
    payload = {
        "models": [
            {"name": "tinyllama:latest"},
            {"name": "nomic-embed-text:latest"},
        ]
    }

    with patch(
        "scholaragent.llm.ollama_client.urlopen",
        return_value=FakeResponse(payload),
    ):
        assert list_models() == [
            "tinyllama:latest",
            "nomic-embed-text:latest",
        ]


def test_generate_returns_clean_text() -> None:
    """Generated text is returned without surrounding whitespace."""
    with patch(
        "scholaragent.llm.ollama_client.urlopen",
        return_value=FakeResponse({"response": "  READY  "}),
    ):
        assert generate("Reply with READY") == "READY"


def test_generate_rejects_empty_prompt() -> None:
    """An empty prompt is rejected before contacting Ollama."""
    with pytest.raises(ValueError, match="must not be empty"):
        generate("   ")


def test_connection_error_is_wrapped() -> None:
    """Network failures become a clear ScholarAgent exception."""
    with patch(
        "scholaragent.llm.ollama_client.urlopen",
        side_effect=URLError("connection refused"),
    ):
        with pytest.raises(OllamaClientError, match="Could not connect"):
            list_models()
