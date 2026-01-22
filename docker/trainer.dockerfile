FROM python:3.12-slim

# Install only essential system dependencies and clean up in same layer
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy only requirements first for better caching
COPY requirements.txt .

# Install PyTorch packages with CUDA support for training
RUN pip install --no-cache-dir torch==2.6.0 torchvision>=0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install remaining requirements and clean up
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy only necessary application files (not all files)
COPY src/ ./src/
COPY pyproject.toml ./
COPY .dvc/ ./.dvc/
COPY data/*.dvc ./data/

# Install the package itself and clean up
RUN pip install --no-cache-dir -e . \
    && pip cache purge

# Initialize git repository for DVC (always do this since .git is excluded)
RUN git init && \
    git config user.email "docker@container" && \
    git config user.name "Docker Container" && \
    git add -A && \
    git commit -m "Initial commit for DVC"

ENV PYTHONPATH=/app/src

# Create entrypoint script that pulls data before training
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Pulling data with DVC..."\n\
dvc pull\n\
echo "Starting training..."\n\
exec python src/ai_real_image_classification/train.py "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
