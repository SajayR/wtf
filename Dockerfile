FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install uv \
    && uv pip install -r requirements.txt

COPY api ./api
COPY ml ./ml
COPY data ./data
COPY pipelines ./pipelines
COPY scripts ./scripts

RUN python -m pipelines.compile || true

EXPOSE 6000

CMD ["gunicorn", "api.app:create_app", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "600"]
