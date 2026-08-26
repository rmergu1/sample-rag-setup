"""Simple, dependency-free paragraph-aware chunker with overlap."""


def split_text(text: str, chunk_size: int = 800, overlap: int = 100):
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                # paragraph itself is too big -- hard-split it
                step = max(chunk_size - overlap, 1)
                for i in range(0, len(para), step):
                    chunks.append(para[i:i + chunk_size])
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) < 2:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prefix = chunks[i - 1][-overlap:]
        overlapped.append((prefix + "\n" + chunks[i]).strip())

    return overlapped
