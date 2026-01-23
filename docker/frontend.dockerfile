FROM python:3.11-slim

# Install system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git curl && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    streamlit \
    requests \
    pillow \
    pandas \
    google-cloud-run

# Copy your source code
COPY src/ ./src/

# Expose the port Cloud Run uses
EXPOSE 8080

# Use shell form so $PORT is expanded
ENTRYPOINT ["sh", "-c", "streamlit run src/ai_real_image_classification/frontend.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false"]
