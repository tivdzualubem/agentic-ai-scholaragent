"""Tests for official-source acquisition and normalization."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from scholaragent.ingestion import (
    SourceIngestionError,
    extract_html,
    fetch_source,
    load_source_snapshot,
    save_source_snapshot,
)


HTML = b"""
<!doctype html>
<html>
<head>
  <title> Example Scholarship 2027 </title>
  <style>.hidden { display: none; }</style>
  <script>window.secret = "ignore";</script>
</head>
<body>
  <main>
    <h1>Example Scholarship</h1>
    <p>International master's applicants may apply.</p>
    <p>The deadline is 15 January 2027.</p>
  </main>
</body>
</html>
"""


def test_html_extraction_removes_non_content() -> None:
    """HTML extraction should preserve useful official text."""
    title, text = extract_html(HTML)

    assert title == "Example Scholarship 2027"
    assert "Example Scholarship" in text
    assert "International master's applicants" in text
    assert "15 January 2027" in text

    assert "window.secret" not in text
    assert "display: none" not in text


def test_fetch_source_preserves_provenance() -> None:
    """Fetched HTML should produce a validated source snapshot."""
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={
                "content-type": "text/html; charset=utf-8",
            },
            content=HTML,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(
        transport=transport,
    ) as client:
        document = fetch_source(
            "https://scholarships.example.edu/program",
            checked_on=date(2026, 6, 28),
            client=client,
        )

    assert document.source_type == "html"
    assert document.title == "Example Scholarship 2027"
    assert document.source_last_checked == date(
        2026,
        6,
        28,
    )
    assert document.byte_size == len(HTML)
    assert len(document.content_sha256) == 64
    assert str(document.source_url).startswith(
        "https://scholarships.example.edu/"
    )


def test_fetch_source_rejects_oversized_content() -> None:
    """The configured source-size limit must be enforced."""
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/plain"},
            content=b"x" * 100,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(
        transport=transport,
    ) as client:
        with pytest.raises(
            SourceIngestionError,
            match="size limit",
        ):
            fetch_source(
                "https://example.edu/large-source.txt",
                max_bytes=10,
                client=client,
            )


def test_source_snapshot_round_trip(
    tmp_path: Path,
) -> None:
    """Saved provenance snapshots should load unchanged."""
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/html"},
            content=HTML,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(
        transport=transport,
    ) as client:
        document = fetch_source(
            "https://example.edu/scholarship",
            checked_on=date(2026, 6, 28),
            client=client,
        )

    path = save_source_snapshot(
        document,
        tmp_path / "snapshot.json",
    )

    loaded = load_source_snapshot(path)

    assert loaded == document


def test_fetch_source_rejects_non_http_url() -> None:
    """Local-file and unsupported schemes must be rejected."""
    with pytest.raises(
        ValueError,
        match="HTTP or HTTPS",
    ):
        fetch_source(
            "file:///etc/passwd"
        )
