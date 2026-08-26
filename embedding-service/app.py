import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# BAAI/bge-small-en-v1.5 (384 dims) is the default: fast on CPU, strong quality.
# Swap to BAAI/bge-base-en-v1.5 (768 dims) for a quality bump at ~2-3x the latency
# -- if you do, also change VECTOR(384) to VECTOR(768) in db/init.sql.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# BGE models recommend prefixing *queries* (not passages) with this instruction
# for retrieval tasks -- meaningfully improves recall.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

app = FastAPI(title="Embedding Service", version="1.0")
model = SentenceTransformer(MODEL_NAME, device="cpu")


class EmbedRequest(BaseModel):
    texts: List[str]
    type: str = "passage"  # "passage" (documents) or "query" (user questions)


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    texts = req.texts
    if req.type == "query":
        texts = [QUERY_PREFIX + t for t in texts]

    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = [v.tolist() for v in vectors]

    return EmbedResponse(
        embeddings=embeddings,
        model=MODEL_NAME,
        dimensions=len(embeddings[0]) if embeddings else 0,
    )
