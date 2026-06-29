FROM python:3.12-slim

ARG OLLAMA_VERSION=0.24.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:/usr/local/bin:${PATH} \
    PORT=7860 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    OLLAMA_MODELS=/home/user/.ollama/models \
    OLLAMA_MODEL=tinyllama:latest \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=2 \
    OLLAMA_KEEP_ALIVE=5m \
    OLLAMA_CONTEXT_LENGTH=2048 \
    OLLAMA_NO_CLOUD=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tini \
        zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh \
        | OLLAMA_VERSION="${OLLAMA_VERSION}" sh \
    && ollama --version

RUN useradd --create-home --uid 1000 user

WORKDIR /home/user/app

COPY --chown=user:user . /home/user/app

USER user

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e '.[demo]'

RUN set -eux; \
    mkdir -p "${OLLAMA_MODELS}"; \
    ollama serve >/tmp/ollama-build.log 2>&1 & \
    server_pid=$!; \
    ready=0; \
    for attempt in $(seq 1 120); do \
        if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then \
            ready=1; \
            break; \
        fi; \
        sleep 1; \
    done; \
    if [ "${ready}" -ne 1 ]; then \
        cat /tmp/ollama-build.log; \
        exit 1; \
    fi; \
    ollama pull tinyllama:latest; \
    ollama pull nomic-embed-text:latest; \
    ollama list; \
    kill "${server_pid}" 2>/dev/null || true; \
    wait "${server_pid}" 2>/dev/null || true; \
    rm -f /tmp/ollama-build.log

RUN chmod 755 /home/user/app/entrypoint.sh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://127.0.0.1:7860/_stcore/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/home/user/app/entrypoint.sh"]
