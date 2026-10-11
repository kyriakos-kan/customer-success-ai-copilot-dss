# One Dockerfile, three targets: api, pipeline, frontend.
# Each target installs only its own dependencies, so the frontend image
# never carries xgboost/shap and the api image never carries streamlit.

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements/base.txt requirements/base.txt
RUN pip install -r requirements/base.txt

# ---------- pipeline (monitoring -> analyst -> RAG -> NBA -> writer) ----------
FROM base AS pipeline
# libgomp is required by xgboost / lightgbm at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements/pipeline.txt requirements/pipeline.txt
RUN pip install -r requirements/pipeline.txt
COPY src/ src/
COPY playbook/ playbook/
RUN mkdir -p /models && chown app:app /models
USER app
CMD ["python", "-m", "copilot.pipeline.orchestrator"]

# ---------- api (FastAPI) ----------
FROM base AS api
COPY requirements/api.txt requirements/api.txt
RUN pip install -r requirements/api.txt
COPY src/ src/
COPY playbook/ playbook/
USER app
EXPOSE 8000
CMD ["uvicorn", "copilot.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ---------- frontend (Streamlit human review) ----------
FROM python:3.12-slim AS frontend
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY requirements/frontend.txt requirements/frontend.txt
RUN pip install -r requirements/frontend.txt
COPY frontend/ frontend/
USER app
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
