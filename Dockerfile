# lightweight python
FROM python:3.11-slim

WORKDIR /app

# requirements
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements_prod.txt .

# install
RUN pip install --no-cache-dir -r requirements_prod.txt

# copy src and model
COPY src/ src/
COPY models/ models/

# expose port
EXPOSE 8080

# run application
CMD ["uvicorn", "src.ai_real_image_classification.api:app", "--host", "0.0.0.0", "--port", "8080"]