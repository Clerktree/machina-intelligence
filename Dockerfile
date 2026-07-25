FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MACHINA_DB_PATH=/data/machina.db \
    MACHINA_CLASSIFIER_PATH=/app/artifacts/cwru-enhanced/model.joblib \
    MACHINA_RUL_MODEL_PATH=/app/artifacts/rul-cmapss/model.joblib \
    MACHINA_QUALITY_MODEL_PATH=/app/artifacts/ai4i-quality/model.joblib

COPY pyproject.toml README.md ./
COPY src ./src
COPY artifacts ./artifacts
COPY configs ./configs

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[mcp,runtime]"

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "machina_harness.api:app", "--host", "0.0.0.0", "--port", "8000"]
