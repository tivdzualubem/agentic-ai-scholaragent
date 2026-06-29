#!/usr/bin/env bash
set -euo pipefail

OLLAMA_LOG="/tmp/scholaragent-ollama.log"
PORT="${PORT:-7860}"

echo "Starting ScholarAgent full-stack container."
echo "Ollama client installed"
echo "Streamlit port: ${PORT}"

ollama serve >"${OLLAMA_LOG}" 2>&1 &
OLLAMA_PID=$!

cleanup() {
    kill "${OLLAMA_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready=0

for attempt in $(seq 1 120); do
    if curl -fsS \
        http://127.0.0.1:11434/api/tags \
        >/dev/null 2>&1
    then
        ready=1
        break
    fi

    sleep 1
done

if [ "${ready}" -ne 1 ]; then
    echo "ERROR: Ollama did not become ready."
    cat "${OLLAMA_LOG}"
    exit 1
fi

for model in \
    "tinyllama:latest" \
    "nomic-embed-text:latest"
do
    if ! ollama list | grep -Fq "${model}"; then
        echo "Model ${model} is missing; downloading it."
        ollama pull "${model}"
    fi
done

echo "Available Ollama models:"
ollama list

exec streamlit run \
    src/scholaragent/ui/streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port="${PORT}" \
    --server.headless=true \
    --browser.gatherUsageStats=false
