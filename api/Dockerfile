FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY api ./api
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}
