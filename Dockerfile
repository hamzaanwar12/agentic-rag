FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src src
RUN uv sync --frozen --no-dev
RUN useradd -r -u 10001 app && mkdir /data && chown app /data
USER app
ENV RAG_DB=/data/rag.db
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "agentic_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
