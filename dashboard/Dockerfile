FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY dashboard ./dashboard
CMD exec streamlit run dashboard/app.py \
    --server.port=${PORT:-8080} --server.address=0.0.0.0 \
    --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
