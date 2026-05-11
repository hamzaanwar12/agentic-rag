FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src src
RUN uv sync --frozen --no-dev
ENV RAG_DB=/data/rag.db
EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "agentic_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
