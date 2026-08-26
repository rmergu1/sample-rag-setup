import os
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from psycopg2 import pool
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel

from providers import ProviderError, call_llm, resolve_provider_and_model

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")
TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))

COMPANY_NAME = os.getenv("COMPANY_NAME", "FHIREngine")

# Below this cosine-similarity score, we don't trust the retrieved context enough
# to let the LLM answer -- return a canned "not found" reply instead.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

NOT_FOUND_MESSAGE = os.getenv(
    "NOT_FOUND_MESSAGE",
    "Sorry, I don't have that information available at this time. "
    "Please directly reach out to the FH team.",
)

WELCOME_MESSAGE = os.getenv("WELCOME_MESSAGE") or (
    f"👋 Hi, I'm the **{COMPANY_NAME} Assistant**.\n\n"
    "I can help answer questions about **NCCN norms** and **FH (FHIREngine) "
    "application details**, based on our internal documentation.\n\n"
    "**Try asking things like:**\n"
    "- What are the NCCN guidelines for a specific condition?\n"
    "- How does the FH application handle a particular workflow?\n"
    "- Where do I configure X in the FH system?\n\n"
    "Go ahead, ask me anything!"
)

_GREETING_WORDS = {
    "hi", "hii", "hiii", "hiya", "hello", "helo", "hey", "heyy", "heya",
    "yo", "sup", "greetings", "namaste",
}
_GREETING_PHRASES = {
    "good morning", "good afternoon", "good evening", "what's up",
    "whats up", "how are you", "how's it going", "hows it going",
}


def is_greeting(text: str) -> bool:
    """Detect short greeting-only messages so we can return the welcome
    message instead of running retrieval + an LLM call for them."""
    normalized = text.strip().lower().strip("!.,?; ")
    if not normalized:
        return False
    if normalized in _GREETING_WORDS or normalized in _GREETING_PHRASES:
        return True
    words = normalized.split()
    # short message starting with a greeting word, e.g. "hi there", "hey claude"
    if len(words) <= 4 and words[0] in _GREETING_WORDS:
        return True
    return False

DB_DSN = (
    f"host={os.getenv('POSTGRES_HOST', 'db')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'ragdb')} "
    f"user={os.getenv('POSTGRES_USER', 'raguser')} "
    f"password={os.getenv('POSTGRES_PASSWORD', '')}"
)

app = FastAPI(title="RAG API", version="1.0")
db_pool: Optional[pool.SimpleConnectionPool] = None


@app.on_event("startup")
def startup():
    global db_pool
    db_pool = pool.SimpleConnectionPool(1, 10, DB_DSN)
    conn = db_pool.getconn()
    register_vector(conn)
    db_pool.putconn(conn)


@app.on_event("shutdown")
def shutdown():
    if db_pool:
        db_pool.closeall()


class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = None      # e.g. "llama-3.1-8b-instant" or "google/gemini-1.5-flash"
    provider: Optional[str] = None   # "groq" | "google" | "openrouter" -- default: groq
    top_k: Optional[int] = None


class Source(BaseModel):
    source_path: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    provider: str
    model: str
    sources: List[Source]
    retrieved_chunks: int


async def embed_query(text: str) -> List[float]:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{EMBEDDING_SERVICE_URL}/embed", json={"texts": [text], "type": "query"}
        )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def retrieve_chunks(embedding: List[float], top_k: int):
    conn = db_pool.getconn()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.source_path, c.chunk_index, c.content, 1 - (c.embedding <=> %s::vector) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'active'
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )
            return cur.fetchall()
    finally:
        db_pool.putconn(conn)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents WHERE status = 'active'")
            doc_count = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM chunks")
            chunk_count = cur.fetchone()[0]
        return {"documents": doc_count, "chunks": chunk_count}
    finally:
        db_pool.putconn(conn)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "question must not be empty")

    # Greetings ("hi", "hello", ...) never touch retrieval or the LLM --
    # answer with the same canned welcome message every time, via curl or UI.
    if is_greeting(question):
        return QueryResponse(
            answer=WELCOME_MESSAGE, provider="none", model="none", sources=[], retrieved_chunks=0
        )

    top_k = req.top_k or TOP_K_DEFAULT

    try:
        embedding = await embed_query(question)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"embedding service error: {e}")

    rows = retrieve_chunks(embedding, top_k)
    top_score = float(rows[0][3]) if rows else 0.0

    # Low (or no) similarity match -- don't let the LLM guess, return the
    # standard "not found" reply directly. Consistent wording, no API call spent.
    if not rows or top_score < SIMILARITY_THRESHOLD:
        return QueryResponse(
            answer=NOT_FOUND_MESSAGE, provider="none", model="none", sources=[], retrieved_chunks=len(rows)
        )

    try:
        provider, model = resolve_provider_and_model(req.provider, req.model)
    except ProviderError as e:
        raise HTTPException(400, str(e))

    context = "\n\n---\n\n".join(f"[Source: {r[0]}, chunk {r[1]}]\n{r[2]}" for r in rows)
    sources = [Source(source_path=r[0], chunk_index=r[1], score=float(r[3])) for r in rows]

    system_prompt = (
        "You are a helpful assistant answering questions using ONLY the provided context "
        "from internal documents. If the answer genuinely isn't in the context, reply "
        f'exactly: "{NOT_FOUND_MESSAGE}" -- do not make things up. Otherwise be concise '
        "and mention which source file(s) you used."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        answer = await call_llm(provider, model, system_prompt, user_prompt)
    except ProviderError as e:
        raise HTTPException(502, str(e))

    return QueryResponse(
        answer=answer,
        provider=provider,
        model=model,
        sources=sources,
        retrieved_chunks=len(rows),
    )
