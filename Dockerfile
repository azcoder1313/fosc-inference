FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    wget curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Create model dir — custom weights go here when trained
RUN mkdir -p model

# Pre-download YOLOv11n base weights as placeholder
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" || true

ENV PORT=8000
ENV CONFIDENCE_THRESHOLD=0.45
ENV MODEL_PATH=model/fosc_v1.pt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
