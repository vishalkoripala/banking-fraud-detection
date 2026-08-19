# Banking Fraud Detection API
# Build:  docker build -t fraud-api .
# Run:    docker run -p 8000:8000 fraud-api
# Docs:   http://localhost:8000/docs

FROM python:3.12-slim

WORKDIR /app

# System deps: none beyond what pip needs for this stack (no compilers required
# for pandas/scikit-learn/shap wheels on standard manylinux platforms).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first, separately from app code, so `docker build` only
# re-runs the (slow) dependency install when requirements.txt actually changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY transformers.py pipeline.py evaluate.py explain.py demo.py api.py ./

# Trained artifacts (model + metadata + SHAP background sample). These are
# produced by `python train_final_model.py` -- rebuild and re-copy this
# directory to deploy a newly trained model; the API and Dockerfile don't
# need to change.
COPY model/ ./model/

# Run as a non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${PORT:-8000}/health || exit 1'

# Shell form (not exec/JSON array) so ${PORT:-8000} actually expands: most
# PaaS hosts (Render, Railway, etc.) inject PORT at runtime and expect the
# container to bind to it; this falls back to 8000 for local/Compose use
# where PORT is unset.
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
