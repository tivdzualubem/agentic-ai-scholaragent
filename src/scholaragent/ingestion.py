"""Acquisition and normalization of official scholarship sources."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
)
from pypdf import PdfReader


DEFAULT_MAX_SOURCE_BYTES = 5_000_000
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_USER_AGENT = (
    "ScholarAgent/0.1 "
    "(academic scholarship-research prototype)"
)


class SourceIngestionError(RuntimeError):
    """Raised when an official source cannot be safely ingested."""


class SourceDocument(BaseModel):
    """Normalized snapshot of one official source."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    source_url: HttpUrl
    source_type: Literal["html", "pdf", "text"]
    title: str | None = None
    text: str = Field(min_length=1)
    source_last_checked: date
    content_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$"
    )
    byte_size: int = Field(ge=1)


def _normalize_lines(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    normalized_lines: list[str] = []
    previous_line: str | None = None

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()

        if not line:
            continue

        if line == previous_line:
            continue

        normalized_lines.append(line)
        previous_line = line

    return "\n".join(normalized_lines)


def extract_html(
    content: bytes,
) -> tuple[str | None, str]:
    """Extract readable text and title from an HTML document."""
    soup = BeautifulSoup(content, "html.parser")

    title = None

    if soup.title is not None:
        title_text = soup.title.get_text(
            " ",
            strip=True,
        )
        title = (
            re.sub(r"\s+", " ", title_text).strip()
            or None
        )

    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "canvas",
            "template",
        ]
    ):
        element.decompose()

    content_root = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    text = _normalize_lines(
        content_root.get_text("\n", strip=True)
    )

    if not text:
        raise SourceIngestionError(
            "The HTML source contains no readable text."
        )

    return title, text


def extract_pdf(
    content: bytes,
) -> tuple[str | None, str]:
    """Extract text and metadata title from a PDF document."""
    try:
        reader = PdfReader(BytesIO(content))
    except Exception as exc:
        raise SourceIngestionError(
            "The PDF source could not be parsed."
        ) from exc

    pages: list[str] = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise SourceIngestionError(
                f"Could not extract PDF page {page_number}."
            ) from exc

        normalized = _normalize_lines(page_text)

        if normalized:
            pages.append(normalized)

    text = "\n\n".join(pages)

    if not text:
        raise SourceIngestionError(
            "The PDF source contains no extractable text."
        )

    title = None
    metadata = reader.metadata

    if metadata is not None:
        raw_title = getattr(metadata, "title", None)

        if isinstance(raw_title, str):
            title = (
                re.sub(r"\s+", " ", raw_title).strip()
                or None
            )

    return title, text


def extract_plain_text(
    content: bytes,
) -> tuple[None, str]:
    """Decode and normalize a plain-text source."""
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError:
        decoded = content.decode(
            "utf-8",
            errors="replace",
        )

    text = _normalize_lines(decoded)

    if not text:
        raise SourceIngestionError(
            "The text source contains no readable content."
        )

    return None, text


def _identify_source_type(
    *,
    content_type: str,
    url: str,
) -> Literal["html", "pdf", "text"]:
    """Resolve the supported source format."""
    normalized_type = (
        content_type.split(";", maxsplit=1)[0]
        .strip()
        .casefold()
    )

    if normalized_type in {
        "text/html",
        "application/xhtml+xml",
    }:
        return "html"

    if normalized_type == "application/pdf":
        return "pdf"

    if normalized_type.startswith("text/plain"):
        return "text"

    if urlparse(url).path.casefold().endswith(".pdf"):
        return "pdf"

    raise SourceIngestionError(
        "Unsupported source content type: "
        f"{content_type or 'missing'}."
    )


def _validate_url(url: str) -> None:
    """Allow only normal HTTP and HTTPS official sources."""
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            "Source URL must use HTTP or HTTPS."
        )

    if not parsed.netloc:
        raise ValueError(
            "Source URL must include a hostname."
        )


def fetch_source(
    url: str,
    *,
    checked_on: date | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    client: httpx.Client | None = None,
) -> SourceDocument:
    """Fetch, validate, extract, and fingerprint an official source."""
    _validate_url(url)

    if timeout <= 0:
        raise ValueError(
            "Source timeout must be positive."
        )

    if max_bytes < 1:
        raise ValueError(
            "Maximum source size must be positive."
        )

    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/pdf,text/plain;q=0.9,*/*;q=0.1"
        ),
    }

    owns_client = client is None

    if client is None:
        client = httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers=headers,
        )

    try:
        response = client.get(
            url,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SourceIngestionError(
            f"Could not fetch official source: {exc}"
        ) from exc
    finally:
        if owns_client:
            client.close()

    declared_length = response.headers.get(
        "content-length"
    )

    if declared_length is not None:
        try:
            declared_bytes = int(declared_length)
        except ValueError:
            declared_bytes = None

        if (
            declared_bytes is not None
            and declared_bytes > max_bytes
        ):
            raise SourceIngestionError(
                "Official source exceeds the configured "
                "size limit."
            )

    content = response.content

    if not content:
        raise SourceIngestionError(
            "Official source returned no content."
        )

    if len(content) > max_bytes:
        raise SourceIngestionError(
            "Official source exceeds the configured "
            "size limit."
        )

    content_type = response.headers.get(
        "content-type",
        "",
    )

    source_type = _identify_source_type(
        content_type=content_type,
        url=str(response.url),
    )

    if source_type == "html":
        title, text = extract_html(content)
    elif source_type == "pdf":
        title, text = extract_pdf(content)
    else:
        title, text = extract_plain_text(content)

    return SourceDocument(
        source_url=str(response.url),
        source_type=source_type,
        title=title,
        text=text,
        source_last_checked=(
            checked_on or date.today()
        ),
        content_sha256=hashlib.sha256(
            content
        ).hexdigest(),
        byte_size=len(content),
    )


def save_source_snapshot(
    document: SourceDocument,
    path: str | Path,
) -> Path:
    """Save a normalized source snapshot as JSON."""
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        document.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return output_path


def load_source_snapshot(
    path: str | Path,
) -> SourceDocument:
    """Load and validate a saved source snapshot."""
    snapshot_path = Path(path)

    if not snapshot_path.is_file():
        raise SourceIngestionError(
            f"Source snapshot does not exist: "
            f"{snapshot_path}"
        )

    try:
        raw_data = json.loads(
            snapshot_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise SourceIngestionError(
            "Source snapshot is not valid JSON."
        ) from exc

    return SourceDocument.model_validate(raw_data)
