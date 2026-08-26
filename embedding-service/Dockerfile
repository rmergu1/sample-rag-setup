FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch explicitly to avoid pulling multi-GB CUDA wheels
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
# Cache the model inside the image build so first request isn't slow
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
