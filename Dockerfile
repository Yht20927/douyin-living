# CPU-only development image: recording + rule-based pipeline.
# Build with AI dependencies (large) via:
#   docker build --build-arg INSTALL_AI=1 -t douyin-live:cpu-ai .
FROM python:3.11-slim

ARG INSTALL_AI=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        nodejs \
        npm \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-ai.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_AI" = "1" ]; then pip install --no-cache-dir -r requirements-ai.txt; fi

COPY scripts/package.json scripts/package-lock.json ./scripts/
RUN npm --prefix scripts ci

COPY . .

# Model files are optional and should normally be mounted:
#   docker run -v $PWD/models:/app/models ...
RUN mkdir -p data models

ENTRYPOINT ["python", "-m", "src.controller"]
CMD ["--help"]
