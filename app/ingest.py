import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.config import MODEL_NAME, DATA_DIR
from app.pdf_extraction import extract_pages
from app.chunking import chunk_text
from app.indexing import build_index, save_index


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app.ingest <path_to_pdf_folder>")
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

    all_metadata = []

    for pdf_path in pdf_files:
        chunk_index = 0
        total_chars = 0

        try:
            pages = extract_pages(pdf_path)
        except Exception as e:
            print(f"ERROR: failed to process {pdf_path.name} ({e}). Skipping.")
            continue

        for page_num, text in pages:
            total_chars += len(text)
            for chunk in chunk_text(text):
                all_metadata.append({
                    "document_name": pdf_path.name,
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                    "text": chunk,
                })
                chunk_index += 1

        if total_chars == 0:
            print(f"WARNING: {pdf_path.name} has no extractable text "
                  f"(likely scanned/image-only). Skipping OCR.")
        else:
            print(f"{pdf_path.name}: {chunk_index} chunks")

    if not all_metadata:
        print("No chunks to embed. Exiting.")
        sys.exit(1)

    print(f"\nEmbedding {len(all_metadata)} chunks...")
    index = build_index(model, all_metadata)

    save_index(index, all_metadata, MODEL_NAME, DATA_DIR)
    print(f"\nIndexed {index.ntotal} chunks from {len(pdf_files)} PDF(s). Saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()