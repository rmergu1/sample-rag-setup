# Internal Docs RAG (free-tier, Docker Compose)

A self-contained RAG stack: pgvector for storage/retrieval, a local CPU embedding
model, and a `/query` API that calls out to a free-tier LLM (Groq / Gemini /
OpenRouter) to generate the final answer.

> This project also ships `podman-compose.yaml` for Podman users. On WSL2,
> Docker Compose (`docker-compose.yml`, used below) is the smoother path —
> Podman 3.x on WSL has known gaps in rootless networking (container DNS,
> healthchecks, CNI) that Docker's embedded DNS and native healthcheck support
> don't have. Swap `docker compose` for `podman-compose -f podman-compose.yaml`
> if you prefer Podman and it's working for you.

```
┌─────────────┐     ┌────────────────────┐     ┌─────────────┐
│  ingest.py  │────▶│  embedding-service  │◀────│   rag-api   │
│ (CLI, run   │     │  (bge-small-en-v1.5 │     │ (/query)    │───▶ Groq / Gemini /
│  on demand) │     │   FastAPI, CPU)     │     │             │     OpenRouter
└──────┬──────┘     └─────────────────────┘     └──────┬──────┘
       │                                                │
       └───────────────────▶  db (pgvector)  ◀──────────┘
```

## Services

| Service            | Purpose                                                       | Port |
|---------------------|----------------------------------------------------------------|------|
| `db`                | Postgres 16 + pgvector, stores documents/chunks/embeddings     | 5432 |
| `embedding-service` | Local CPU embedding model (bge-small-en-v1.5, 384 dims)        | 8001 |
| `rag-api`           | `/query` endpoint: embed → retrieve → call an LLM → return JSON| 8080 |
| `frontend`          | Lightweight Apache-served chat UI, proxies `/api/*` → rag-api  | 8090 |
| `ingest`            | CLI, run on demand (not a long-running service)                 | -    |

### Chat UI

Open **http://localhost:8090** in a browser. It's a single static HTML file
(no build step, no framework, no external requests) served by Apache, which
proxies `/api/*` to `rag-api` on the same origin -- so there's no CORS setup
needed. It shows a welcome message with sample prompts on load, and renders
answers/sources from the same `/query` endpoint `curl` uses.

### Greeting + "not found" behaviour (server-side, applies to curl and UI alike)

- A short greeting-only message ("hi", "hello", "hey there", ...) skips
  retrieval and the LLM entirely and returns a canned welcome message with
  sample prompts. Configure via `COMPANY_NAME` / `WELCOME_MESSAGE` in `.env`.
- If the best retrieved chunk's cosine similarity is below `SIMILARITY_THRESHOLD`
  (default `0.35`) -- or nothing was retrieved at all -- `/query` returns a
  fixed reply (`NOT_FOUND_MESSAGE` in `.env`) directly, without spending an
  LLM call. The system prompt also tells the LLM to use the exact same
  wording if it genuinely can't answer from context that did pass the
  threshold, so the reply is consistent either way.
- Both behaviors live in `rag-api`, not the frontend, so `curl -X POST
  http://localhost:8080/query -d '{"question": "hi"}'` gets identical
  treatment to typing "hi" in the chat UI.

## 1. Setup

```bash
cp .env.example .env
# edit .env: set POSTGRES_PASSWORD, and at least one of
# GROQ_API_KEY / GOOGLE_API_KEY / OPENROUTER_API_KEY

docker compose up -d --build db embedding-service rag-api
```

Wait for health checks:

```bash
docker compose ps
curl http://localhost:8080/health
curl http://localhost:8001/health
```

## 2. Ingest documents

Put your files under `./documents/` (or point at any absolute path mounted
into the `ingest` container).

```bash
# ingest a whole folder, recursively (pdf, txt, log, html, htm, png, jpg, jpeg)
docker compose run --rm ingest --action ingest --path /data

# ingest a single file
docker compose run --rm ingest --action ingest --path /data/handbook.pdf

# re-running ingest is safe: unchanged files (same sha256) are skipped,
# changed files have their chunks replaced automatically

# delete a single file's chunks
docker compose run --rm ingest --action delete --path /data/handbook.pdf

# delete everything under a folder
docker compose run --rm ingest --action delete --path /data/old_docs
```

PDFs get both native text extraction **and** OCR of any images embedded in
the PDF pages (diagrams, screenshots, scanned text) via Tesseract, so text
trapped in images still becomes searchable.

Check what's ingested:

```bash
curl http://localhost:8080/stats
```

## 3. Query

Default model (uses `DEFAULT_PROVIDER=groq` and `GROQ_DEFAULT_MODEL`):

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I reset a user password in the app?"}'
```

Pick a specific provider/model per request:

```bash
# Groq, explicit model
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the API rate limits?", "provider": "groq", "model": "llama-3.1-8b-instant"}'

# Gemini
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the API rate limits?", "provider": "google", "model": "gemini-1.5-flash"}'

# OpenRouter free model
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the API rate limits?", "provider": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:free"}'

# Shorthand: "provider/model" in the model field also works (except openrouter
# model names, which contain slashes themselves -- use explicit "provider" for those)
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the API rate limits?", "model": "google/gemini-1.5-flash"}'
```

Response shape:

```json
{
  "answer": "...",
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "sources": [
    {"source_path": "/data/handbook.pdf", "chunk_index": 3, "score": 0.81}
  ],
  "retrieved_chunks": 5
}
```

## Design notes / things you may want to change

- **Embedding model**: `BAAI/bge-small-en-v1.5` (384 dims) is the default --
  best speed/quality tradeoff on CPU. `bge-base-en-v1.5` (768 dims) is a
  quality step up at ~2-3x latency; if you switch, also change `VECTOR(384)`
  to `VECTOR(768)` in `db/init.sql` and re-ingest everything (dimensions
  aren't interchangeable). The model is baked into the image at build time so
  first requests aren't slow downloading weights.
- **Index**: uses pgvector's `hnsw` index (cosine distance), which builds
  incrementally as you ingest -- no periodic retraining needed like `ivfflat`.
- **Change detection**: each document's sha256 is stored; re-ingesting an
  unchanged file is a no-op, a changed file gets its chunks replaced.
- **Chunking**: paragraph-aware splitter, ~800 chars with 100 char overlap
  (tune via `CHUNK_SIZE`/`CHUNK_OVERLAP` in `.env`). This affects answer
  quality more than model choice -- worth tuning for your document structure.
- **Data sensitivity**: `/query` sends retrieved chunks to whichever external
  LLM API you configure (Groq/Google/OpenRouter). If the source documents are
  sensitive pre-client-release, consider swapping the LLM call for a
  self-hosted model (e.g. Ollama running Llama 3.1 8B) instead of a hosted
  free tier -- happy to wire that in as a fourth provider if useful.
- **Scaling**: `db_pool` in `rag-api` is a small connection pool; for real
  production load put a reverse proxy / rate limiting in front of `rag-api`
  and consider read replicas for `db` -- out of scope for this base setup.
