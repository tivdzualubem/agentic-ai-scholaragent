"""Command-line interface for official-source ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from scholaragent.ingestion import (
    DEFAULT_MAX_SOURCE_BYTES,
    fetch_source,
    save_source_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the official-source ingestion parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and normalize an official scholarship "
            "webpage, PDF, or text source."
        )
    )

    parser.add_argument("url")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_SOURCE_BYTES,
    )

    return parser


def main() -> None:
    """Fetch an official source and save its snapshot."""
    args = build_parser().parse_args()

    document = fetch_source(
        args.url,
        timeout=args.timeout,
        max_bytes=args.max_bytes,
    )

    output = save_source_snapshot(
        document,
        args.output,
    )

    print("Source URL:", document.source_url)
    print("Source type:", document.source_type)
    print("Title:", document.title or "n/a")
    print("Checked:", document.source_last_checked)
    print("Bytes:", document.byte_size)
    print("SHA-256:", document.content_sha256)
    print("Extracted characters:", len(document.text))
    print("Saved snapshot:", output)


if __name__ == "__main__":
    main()
