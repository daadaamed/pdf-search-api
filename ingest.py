import sys
import json
import pdfplumber
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_DIR = Path("data")


def chunk_text(text, size=500, overlap=50):
    if size <= overlap:
        raise ValueError(f"chunk size ({size}) must be greater than overlap ({overlap})")

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
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
    embeddings = embeddings.astype("float32")
    assert embeddings.shape[0] == len(all_metadata), "embeddings and metadata are out of sync!"

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    DATA_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(DATA_DIR / "index.faiss"))
    with open(DATA_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    index_info = {
        "model_name": MODEL_NAME,
        "dimension": dimension,
        "num_vectors": index.ntotal,
    }
    with open(DATA_DIR / "index_info.json", "w", encoding="utf-8") as f:
        json.dump(index_info, f, ensure_ascii=False, indent=2)

    print(f"\nIndexed {index.ntotal} vectors.")
    print(f"Saved to {DATA_DIR / 'index.faiss'}, {DATA_DIR / 'metadata.json'}, "
          f"and {DATA_DIR / 'index_info.json'}")


if __name__ == "__main__":
    main()