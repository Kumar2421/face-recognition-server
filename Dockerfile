FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /media/frigate/facefolder /media/frigate /models
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY ui_page.py /app/ui_page.py
COPY events_store.py /app/events_store.py
COPY config_loader.py /app/config_loader.py
COPY sftp_poller.py /app/sftp_poller.py
COPY config.yaml /app/config.yaml
COPY quality /app/quality
COPY embedders /app/embedders

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
