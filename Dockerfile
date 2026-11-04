FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/download_corpus.py ./scripts/download_corpus.py

RUN uv pip install --no-cache --system .

RUN python scripts/download_corpus.py && \
    findex index data/corpus --out /app/index.bin

FROM python:3.12-slim-bookworm AS runtime

RUN useradd -m -u 1000 appuser

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/index.bin /app/index.bin
COPY --from=builder /app/src /app/src

RUN chown -R appuser:appuser /app
USER appuser

ENV INDEX_PATH=/app/index.bin
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "findex serve --host 0.0.0.0 --port ${PORT:-8000} --index ${INDEX_PATH}"]