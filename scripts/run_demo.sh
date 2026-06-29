#!/usr/bin/env bash
set -euo pipefail

ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.." &&
    pwd
)"

cd "$ROOT"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv was not found."
    exit 1
fi

source .venv/bin/activate

if ! python -c "import streamlit" \
    >/dev/null 2>&1
then
    echo "ERROR: Streamlit is not installed."
    echo "Run: python -m pip install -e '.[demo]'"
    exit 1
fi

PORT="${PORT:-8501}"

echo "Starting ScholarAgent..."
echo "Open http://localhost:${PORT}"
echo
echo "Most reliable presentation configuration:"
echo "  Execution: Fast verified demo"
echo "  Retriever: BM25-only offline mode"
echo
echo "Hybrid mode requires nomic-embed-text:latest."
echo "Full mode also requires tinyllama:latest."
echo

exec python -m streamlit run \
    src/scholaragent/ui/streamlit_app.py \
    --server.address 127.0.0.1 \
    --server.port "$PORT" \
    --server.headless true
