-- RAG vector store schema
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source document (pdf, txt, log, html, image)
CREATE TABLE IF NOT EXISTS documents (
    id               SERIAL PRIMARY KEY,
    source_path      TEXT NOT NULL UNIQUE,
    file_name        TEXT NOT NULL,
    file_type        TEXT NOT NULL,
    file_hash        TEXT NOT NULL,           -- sha256, used to detect changes for re-ingest
    file_size_bytes  BIGINT,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    status           TEXT NOT NULL DEFAULT 'active',   -- active | deleted
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents (source_path);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents (status);

-- One row per chunk of a document, with its embedding
-- 384 dims = BAAI/bge-small-en-v1.5. If you switch to bge-base-en-v1.5 (768 dims)
-- or another model, update this column's dimension and rebuild the index.
CREATE TABLE IF NOT EXISTS chunks (
    id            SERIAL PRIMARY KEY,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    content       TEXT NOT NULL,
    token_count   INTEGER,
    embedding     VECTOR(384),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);

-- HNSW index for fast approximate nearest-neighbour cosine search.
-- Builds incrementally as rows are inserted (unlike ivfflat, no retraining needed).
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Audit trail of every ingest/update/delete action run by the ingest script
CREATE TABLE IF NOT EXISTS ingestion_log (
    id          SERIAL PRIMARY KEY,
    source_path TEXT NOT NULL,
    action      TEXT NOT NULL,   -- ingest | update | delete
    status      TEXT NOT NULL,   -- success | skipped | failed
    message     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_log_source_path ON ingestion_log (source_path);
