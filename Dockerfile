FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data

RUN pip install --upgrade pip \
    && pip install ".[ml]"

RUN python -m job_ml.pipeline \
    --data data/raw/sample_jobs.csv \
    --output-dir artifacts

EXPOSE 8000

CMD ["uvicorn", "job_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
