FROM python:3.11-slim

# Create non-root user
RUN groupadd -r stock4u && useradd -r -g stock4u stock4u

# Install security updates and required packages
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R stock4u:stock4u /app \
    && chmod -R 755 /app \
    && chmod -R 600 /app/*.key /app/*.pem 2>/dev/null || true

# Security environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LEARNING_SCHED_ENABLED=1 \
    RATE_LIMIT_PER_MIN=60 \
    # Security hardening
    PYTHONHASHSEED=random \
    # Disable Python bytecode generation
    PYTHONOPTIMIZE=1

# Switch to non-root user
USER stock4u

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]


