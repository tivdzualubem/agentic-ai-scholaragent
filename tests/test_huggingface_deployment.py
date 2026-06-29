from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hugging_face_deployment_files_exist() -> None:
    for name in (
        "Dockerfile",
        "entrypoint.sh",
        ".dockerignore",
    ):
        path = ROOT / name
        assert path.is_file()
        assert path.stat().st_size > 0


def test_dockerfile_packages_complete_stack() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    required = (
        "FROM python:3.12-slim",
        "OLLAMA_VERSION=0.24.0",
        "ollama pull tinyllama:latest",
        "ollama pull nomic-embed-text:latest",
        "python -m pip install --no-cache-dir -e '.[demo]'",
        "EXPOSE 7860",
        "USER user",
        "HEALTHCHECK",
        'ENTRYPOINT ["/usr/bin/tini"',
    )

    for value in required:
        assert value in text


def test_entrypoint_starts_both_services() -> None:
    text = (ROOT / "entrypoint.sh").read_text(
        encoding="utf-8"
    )

    assert "ollama serve" in text
    assert "streamlit run" in text
    assert "0.0.0.0" in text
    assert "tinyllama:latest" in text
    assert "nomic-embed-text:latest" in text


def test_space_readme_metadata() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert text.startswith("---\n")
    assert "sdk: docker" in text
    assert "app_port: 7860" in text
    assert "suggested_hardware: cpu-basic" in text
