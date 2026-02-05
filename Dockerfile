# =============================================================================
# Dockerfile for RAG API Service
# Based on: https://github.com/joshi-sharat/local-rag-system
# =============================================================================

# Use Python 3.12 slim image for smaller footprint
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies required for:
# - APIs (fastapi)
# - APIs processing (pydantic ?)
# - App runner (uvicorn ?)
# - Build tools for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    curl \
    procps \
    net-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy only requirements first (for Docker layer caching)
COPY requirements-api.txt requirements.txt


# Copy only the necessary application files
# Core source code
COPY src/ ./src/

COPY api/ ./api/

# Main application entry point
COPY run_service.py .

# Create necessary directories
RUN mkdir -p /app/logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add user's local bin to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Expose APIs default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the RAG API application
CMD ["python", "run_service.py", "--host", "0.0.0.0", "--port", "8080"]