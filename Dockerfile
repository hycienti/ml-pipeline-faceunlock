FROM python:3.12-slim

# opencv needs these shared libraries even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app ./app
COPY src ./src
COPY models ./models

ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/login')"

# One process with several threads: the heartbeat thread and the model are shared inside it.
# Run more containers (not more workers) to scale; they share one database volume.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "300", "--access-logfile", "-", "app:create_app()"]
