import sys
import pdfplumber
from pathlib import Path


def chunk_text(text: str, size: int = 500, overlap: int = 50)-> list[str]:
    if size <= overlap:
        raise ValueError(f"chunk size ({size}) must be greater than overlap ({overlap})")

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf_folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a valid folder")
        sys.exit(1)

    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder}")
        sys.exit(1)

    for pdf_path in pdf_files:
        total_chars = 0
        chunk_index = 0  # resets per document, NOT per page

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    total_chars += len(text)
                    print(f"{pdf_path.name} - page {page_num} - {len(text)} chars")

                    for chunk in chunk_text(text):
                        preview = chunk[:60].replace("\n", " ")
                        print(f"    chunk {chunk_index}: '{preview}' ({len(chunk)} chars)")
                        chunk_index += 1
        except Exception as e:
            print(f"ERROR: failed to process {pdf_path.name} ({e}). Skipping file.")
            continue

        if total_chars == 0:
            print(f"WARNING: {pdf_path.name} yielded 0 chars across all pages "
                  f"(likely a scanned/image-only PDF, no text layer). "
                  f"Skipping OCR — this file will not be searchable.")


if __name__ == "__main__":
    main()