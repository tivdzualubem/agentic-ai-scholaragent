"""Language-model integrations for ScholarAgent."""

from scholaragent.llm.ollama_client import (
    OllamaClientError,
    generate,
    list_models,
)

__all__ = [
    "OllamaClientError",
    "generate",
    "list_models",
]
