"""
Ingest / delete documents in the RAG vector store.

Usage:
  python ingest.py --action ingest --path /data/some_file.pdf
  python ingest.py --action ingest --path /data/docs_folder      # recurses subfolders
  python ingest.py --action delete --path /data/some_file.pdf
  python ingest.py --action delete --path /data/docs_folder      # deletes everything under it

Supported file types: .pdf .txt .log .html .htm .png .jpg .jpeg
For PDFs: extracts native text AND OCRs any images embedded in the PDF pages.
Re-running ingest on an unchanged file (same sha256) is a no-op (skipped).
Re-running ingest on a changed file replaces its chunks (update).
"""
import argparse
import hashlib
import io
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF
import httpx
import psycopg2
import pytesseract
from bs4 import BeautifulSoup
from PIL import Image
from pgvector.psycopg2 import register_vector

from chunking import split_text

SUPPORTED_TEXT_EXTS = {".txt", ".log"}
SUPPORTED_HTML_EXTS = {".html", ".htm"}
SUPPORTED_PDF_EXTS = {".pdf"}
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
ALL_SUPPORTED = SUPPORTED_TEXT_EXTS | SUPPORTED_HTML_EXTS | SUPPORTED_PDF_EXTS | SUPPORTED_IMAGE_EXTS

DB_DSN = (
    f"host={os.getenv('POSTGRES_HOST', 'db')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'ragdb')} "
    f"user={os.getenv('POSTGRES_USER', 'raguser')} "
    f"password={os.getenv('POSTGRES_PASSWORD', '')}"
)
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
EMBED_BATCH_SIZE = 32


# ---------- extraction ----------

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def extract_text_from_txt(path: Path) -> str:
    return path.read_text(errors="ignore")


def extract_text_from_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")
    return soup.get_text(separator="\n")


def extract_text_from_image(path: Path) -> str:
    try:
        return pytesseract.image_to_string(Image.open(path))
    except Exception as e:
        print(f"  [warn] OCR failed for {path}: {e}")
        return ""


def extract_text_from_pdf(path: Path) -> str:
    """Native text per page, plus OCR text from any images embedded in the PDF."""
    text_parts = []
    doc = fitz.open(str(path))
    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        if page_text.strip():
            text_parts.append(page_text)

        for img in page.get_images(full=True):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                pil_img = Image.open(io.BytesIO(base_image["image"]))
                ocr_text = pytesseract.image_to_string(pil_img)
                if ocr_text.strip():
                    text_parts.append(f"[OCR image, page {page_num + 1}]\n{ocr_text}")
            except Exception as e:
                print(f"  [warn] could not OCR an image on page {page_num + 1}: {e}")

    doc.close()
    return "\n\n".join(text_parts)


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SUPPORTED_TEXT_EXTS:
        return extract_text_from_txt(path)
    if ext in SUPPORTED_HTML_EXTS:
        return extract_text_from_html(path)
    if ext in SUPPORTED_PDF_EXTS:
        return extract_text_from_pdf(path)
    if ext in SUPPORTED_IMAGE_EXTS:
        return extract_text_from_image(path)
    raise ValueError(f"Unsupported file type: {ext}")


def get_embeddings(texts):
    all_vectors = []
    with httpx.Client(timeout=120) as client:
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            resp = client.post(f"{EMBEDDING_SERVICE_URL}/embed", json={"texts": batch, "type": "passage"})
            resp.raise_for_status()
            all_vectors.extend(resp.json()["embeddings"])
    return all_vectors


# ---------- db ----------

def get_connection():
    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)
    return conn


def log_action(cur, source_path, action, status, message=""):
    cur.execute(
        "INSERT INTO ingestion_log (source_path, action, status, message) VALUES (%s, %s, %s, %s)",
        (source_path, action, status, (message or "")[:2000]),
    )


def ingest_file(conn, path: Path):
    source_path = str(path.resolve())
    ext = path.suffix.lower()

    with conn.cursor() as cur:
        try:
            file_hash = sha256_of_file(path)
            cur.execute("SELECT id, file_hash FROM documents WHERE source_path = %s", (source_path,))
            existing = cur.fetchone()

            if existing and existing[1] == file_hash:
                print(f"  [skip] unchanged: {source_path}")
                log_action(cur, source_path, "ingest", "skipped", "unchanged hash")
                conn.commit()
                return

            print(f"  [read] {source_path}")
            text = extract_text(path)
            if not text.strip():
                print(f"  [warn] no extractable text, skipping: {source_path}")
                log_action(cur, source_path, "ingest", "failed", "no extractable text")
                conn.commit()
                return

            chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                log_action(cur, source_path, "ingest", "failed", "no chunks produced")
                conn.commit()
                return

            if existing:
                doc_id = existing[0]
                cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                cur.execute(
                    """UPDATE documents SET file_hash=%s, file_size_bytes=%s, updated_at=now(), status='active'
                       WHERE id=%s""",
                    (file_hash, path.stat().st_size, doc_id),
                )
                action = "update"
            else:
                cur.execute(
                    """INSERT INTO documents (source_path, file_name, file_type, file_hash, file_size_bytes)
                       VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                    (source_path, path.name, ext.lstrip("."), file_hash, path.stat().st_size),
                )
                doc_id = cur.fetchone()[0]
                action = "ingest"

            print(f"  [embed] {len(chunks)} chunks")
            vectors = get_embeddings(chunks)

            for idx, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
                cur.execute(
                    """INSERT INTO chunks (document_id, chunk_index, content, token_count, embedding)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (doc_id, idx, chunk_text, len(chunk_text.split()), vector),
                )

            log_action(cur, source_path, action, "success", f"{len(chunks)} chunks")
            conn.commit()
            print(f"  [done] {action}d {source_path} ({len(chunks)} chunks)")

        except Exception as e:
            conn.rollback()
            print(f"  [error] {source_path}: {e}")
            with conn.cursor() as cur2:
                log_action(cur2, source_path, "ingest", "failed", str(e))
                conn.commit()


def walk_and_ingest(input_path: Path):
    conn = get_connection()
    try:
        if input_path.is_file():
            if input_path.suffix.lower() in ALL_SUPPORTED:
                ingest_file(conn, input_path)
            else:
                print(f"Unsupported file type, skipping: {input_path}")
            return

        files = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in ALL_SUPPORTED
        )
        print(f"Found {len(files)} supported files under {input_path}")
        for f in files:
            ingest_file(conn, f)
    finally:
        conn.close()


def delete_source(target: str):
    target_resolved = str(Path(target).resolve())
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, source_path FROM documents WHERE source_path = %s", (target_resolved,))
            exact = cur.fetchall()

            if exact:
                for doc_id, source_path in exact:
                    cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                    log_action(cur, source_path, "delete", "success", "exact match")
                    print(f"  [deleted] {source_path}")
            else:
                cur.execute(
                    "SELECT id, source_path FROM documents WHERE source_path LIKE %s",
                    (target_resolved.rstrip("/") + "/%",),
                )
                prefix_matches = cur.fetchall()
                if not prefix_matches:
                    print(f"No matching documents found for: {target_resolved}")
                    log_action(cur, target_resolved, "delete", "failed", "no match")
                    conn.commit()
                    return
                for doc_id, source_path in prefix_matches:
                    cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
                    log_action(cur, source_path, "delete", "success", "prefix match")
                    print(f"  [deleted] {source_path}")

            conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Ingest or delete documents in the RAG vector store")
    parser.add_argument("--action", required=True, choices=["ingest", "delete"])
    parser.add_argument("--path", required=True, help="File/folder to ingest, or source path to delete")
    args = parser.parse_args()

    if args.action == "ingest":
        input_path = Path(args.path)
        if not input_path.exists():
            print(f"Path does not exist: {input_path}")
            sys.exit(1)
        walk_and_ingest(input_path)
    else:
        delete_source(args.path)


if __name__ == "__main__":
    main()
