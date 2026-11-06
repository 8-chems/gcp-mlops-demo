FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY dashboard ./dashboard
# entrypoint.py fetches secrets.toml from Secret Manager, then exec's Streamlit
CMD ["python", "dashboard/entrypoint.py"]
