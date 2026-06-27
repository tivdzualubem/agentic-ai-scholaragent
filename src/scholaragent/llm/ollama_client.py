"""Minimal Ollama API client using only the Python standard library."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "tinyllama:latest"


class OllamaClientError(RuntimeError):
    """Raised when ScholarAgent cannot communicate with Ollama."""


def _base_url() -> str:
    """Return the configured Ollama base URL without a trailing slash."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _request_json(
    endpoint: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Send a JSON request to Ollama and return the decoded response."""
    url = f"{_base_url()}{endpoint}"

    data = None
    method = "GET"

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"

    request = Request(
        url=url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise OllamaClientError(
            f"Ollama returned HTTP {exc.code}: {details}"
        ) from exc
    except URLError as exc:
        raise OllamaClientError(
            f"Could not connect to Ollama at {_base_url()}: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise OllamaClientError(
            f"Ollama request timed out after {timeout} seconds."
        ) from exc

    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OllamaClientError(
            "Ollama returned a response that was not valid JSON."
        ) from exc

    if not isinstance(result, dict):
        raise OllamaClientError("Ollama returned an unexpected response type.")

    return result


def list_models() -> list[str]:
    """Return the names of models currently installed in Ollama."""
    response = _request_json("/api/tags")
    models = response.get("models", [])

    return [
        item["name"]
        for item in models
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]


def generate(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> str:
    """Generate a non-streaming response from a local Ollama model."""
    if not prompt.strip():
        raise ValueError("The prompt must not be empty.")

    selected_model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    response = _request_json(
        "/api/generate",
        payload={
            "model": selected_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        },
        timeout=timeout,
    )

    generated_text = response.get("response")

    if not isinstance(generated_text, str) or not generated_text.strip():
        raise OllamaClientError(
            "Ollama returned no generated response text."
        )

    return generated_text.strip()
