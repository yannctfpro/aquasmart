# AquaSmart — Dockerfile
# ============================================================
# Multi-stage build to keep the final image lightweight.
# Runs the Streamlit dashboard only. Model artefacts are NOT
# baked into the image — users run the pipeline after the build:
#
#   docker compose up --build
#   docker compose exec aquasmart python src/data/collect_data_v2.py
#   docker compose exec aquasmart python src/data/generate_target_v4.py
#   docker compose exec aquasmart python src/data/preprocess_v4.py
#   docker compose exec aquasmart python src/models/train_v4.py
#
# Once trained, the models land in /app/models (mounted as a volume)
# and the Streamlit app at http://localhost:8501 picks them up.
# ============================================================

# ---- Stage 1: build environment ----
FROM python:3.11-slim AS builder

# Avoid writing .pyc files and enable output flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages required by numpy / scikit-learn wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install dependencies in a dedicated prefix so we can copy them over cleanly
COPY requirements.txt .
RUN pip install --prefix=/install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt


# ---- Stage 2: runtime image ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Streamlit configuration: headless, listen on all interfaces
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy only what the app needs at runtime — not notebooks, not raw data
COPY src/ ./src/
COPY streamlit_app.py .
COPY config/ ./config/

# Create directories that the pipeline will write to
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/reports

# Non-root user for a bit of hardening
RUN useradd --create-home --shell /bin/bash aquasmart && \
    chown -R aquasmart:aquasmart /app
USER aquasmart

EXPOSE 8501

# Healthcheck: Streamlit exposes /_stcore/health on port 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health').read()" || exit 1

CMD ["streamlit", "run", "streamlit_app.py"]
