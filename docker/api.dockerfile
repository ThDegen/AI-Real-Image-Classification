FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install PyTorch packages with CUDA support first
RUN pip install --no-cache-dir torch==2.6.0 torchvision>=0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src

EXPOSE 8000

ENTRYPOINT ["uvicorn", "ai_real_image_classification.api:app", "--host", "0.0.0.0", "--port", "8000"]
