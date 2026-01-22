FROM python:3.12-slim

# Install only essential system dependencies and clean up in same layer
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Install PyTorch CPU-only version (much smaller than CUDA)
# Use CPU version for API inference to reduce image size
RUN pip install --no-cache-dir torch==2.6.0 torchvision>=0.21.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy only necessary application files
COPY src/ ./src/
COPY pyproject.toml ./

# Install the package itself
RUN pip install --no-cache-dir -e . \
    && pip cache purge

ENV PYTHONPATH=/app/src

EXPOSE 8000

ENTRYPOINT ["uvicorn", "ai_real_image_classification.api:app", "--host", "0.0.0.0", "--port", "8000"]
