import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.config import MODEL_NAME, DATA_DIR
from app.pdf_extraction import extract_pages
from app.chunking import chunk_text
from app.indexing import build_index, save_index


@dataclass
class PdfResult:
    pdf_path: Path
    metadata: list[dict]
    total_chars: int = 0
    error: Exception | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into a searchable vector index.")
    parser.add_argument("folder", type=Path, help="Folder containing PDF files to ingest")
    return parser.parse_args()


def die(message: str, code: int = 1) -> None:
    print(message)
    sys.exit(code)


def process_pdf(pdf_path: Path) -> PdfResult:
    """Extract and chunk a single PDF. Never raises; errors are returned, not thrown."""
    try:
        pages = extract_pages(pdf_path)
    except Exception as e:
        return PdfResult(pdf_path, metadata=[], error=e)

    metadata = []
    total_chars = 0
    chunk_index = 0

    for page_num, text in pages:
        total_chars += len(text)
        for chunk in chunk_text(text):
            metadata.append({
                "document_name": pdf_path.name,
                "page_number": page_num,
                "chunk_index": chunk_index,
                "text": chunk,
            })
            chunk_index += 1

    return PdfResult(pdf_path, metadata, total_chars)


def report(result: PdfResult) -> None:
    name = result.pdf_path.name
    if result.error is not None:
        print(f"ERROR: failed to process {name} ({result.error}). Skipping.")
    elif result.total_chars == 0:
        print(f"WARNING: {name} has no extractable text (likely scanned/image-only). Skipping OCR.")
    else:
        print(f"{name}: {len(result.metadata)} chunks")


def main():
    folder = parse_args().folder

    if not folder.is_dir():
        die(f"ERROR: '{folder}' is not a valid folder")

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        die(f"No PDF files found in {folder}")

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    results = [process_pdf(pdf_path) for pdf_path in pdf_files]

    all_metadata = []
    for result in results:
        report(result)
        all_metadata.extend(result.metadata)

    if not all_metadata:
        die("No chunks to embed. Exiting.")

    print(f"\nEmbedding {len(all_metadata)} chunks...")
    index = build_index(model, all_metadata)

    save_index(index, all_metadata, MODEL_NAME, DATA_DIR)
    print(f"\nIndexed {index.ntotal} chunks from {len(pdf_files)} PDF(s). Saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()