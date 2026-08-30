from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def chunk_text(text, size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    """Split text into overlapping character-based chunks, dropping empty/whitespace-only ones."""
    if size <= overlap:
        raise ValueError(f"chunk size ({size}) must be greater than overlap ({overlap})")

    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks