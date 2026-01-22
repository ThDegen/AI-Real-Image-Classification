FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
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

# Initialize git repository for DVC (always do this since .git is excluded)
RUN rm -rf .git && \
    git init && \
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
