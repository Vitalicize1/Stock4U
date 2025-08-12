FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment defaults (override in runtime)
ENV PYTHONUNBUFFERED=1 \
    LEARNING_SCHED_ENABLED=1 \
    RATE_LIMIT_PER_MIN=60

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]


