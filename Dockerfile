FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    g++ \
    build-essential \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN mkdir -p /media/frigate/facefolder /media/frigate /models
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY ui_page.py /app/ui_page.py
COPY events_store.py /app/events_store.py
COPY config_loader.py /app/config_loader.py
COPY sftp_poller.py /app/sftp_poller.py
COPY config.yaml /app/config.yaml
COPY quality /app/quality
COPY embedders /app/embedders

EXPOSE 8000

CMD ["python3.11", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
