import sys
import pdfplumber
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def chunk_text(text, size=500, overlap=50):
    if size <= overlap:
        raise ValueError(f"chunk size ({size}) must be greater than overlap ({overlap})")

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:  # skip whitespace-only chunks
            chunks.append(chunk)
        start += size - overlap
    return chunks


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf_folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"ERROR: '{folder}' is not a valid folder")
        sys.exit(1)

    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder}")
        sys.exit(1)

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    all_chunks = []
    all_metadata = []

    for pdf_path in pdf_files:
        total_chars = 0
        chunk_index = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    total_chars += len(text)
                    print(f"{pdf_path.name} - page {page_num} - {len(text)} chars")

                    for chunk in chunk_text(text):
                        preview = chunk[:60].replace("\n", " ")
                        print(f"    chunk {chunk_index}: '{preview}' ({len(chunk)} chars)")

                        all_chunks.append(chunk)
                        all_metadata.append({
                            "document_name": pdf_path.name,
                            "page_number": page_num,
                            "chunk_index": chunk_index,
                            "text": chunk,
                        })
                        chunk_index += 1
        except Exception as e:
            print(f"ERROR: failed to process {pdf_path.name} ({e}). Skipping file.")
            continue

        if total_chars == 0:
            print(f"WARNING: {pdf_path.name} yielded 0 chars across all pages "
                  f"(likely a scanned/image-only PDF, no text layer). "
                  f"Skipping OCR — this file will not be searchable.")

    if not all_chunks:
        print("\nNo chunks to embed — every PDF failed or yielded no extractable text. Exiting.")
        sys.exit(1)

    print(f"\nEmbedding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Metadata entries: {len(all_metadata)}")
    assert embeddings.shape[0] == len(all_metadata), "embeddings and metadata are out of sync!"


if __name__ == "__main__":
    main()