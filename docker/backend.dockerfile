FROM python:3.12-slim-bookworm

WORKDIR /app

# Ensure logs are sent straight to terminal without buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8080

# Install system dependencies (onnxruntime-slim may need these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    onnxruntime \
    numpy \
    pillow \
    prometheus-client \
    google-cloud-storage \
    python-multipart

COPY src/ ./src/
COPY models/ ./models/

# Use the environment variable PORT provided by Cloud Run
# We use 'sh -c' to ensure the variable is expanded correctly
CMD ["sh", "-c", "uvicorn src.ai_real_image_classification.onnx_fastapi:app --host 0.0.0.0 --port ${PORT}"]
