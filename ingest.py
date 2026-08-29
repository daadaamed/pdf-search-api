import sys
import pdfplumber
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf_folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder}")
        sys.exit(1)

    for pdf_path in pdf_files:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                print(f"{pdf_path.name} - page {page_num} - {len(text)} chars")


if __name__ == "__main__":
    main()