# PDF Search API
 
A small local system to search across a collection of PDF documents by meaning
(semantic search), not just keywords. Built as a CLI ingestion program (PDFs →
text → chunks → embeddings → FAISS index) plus a FastAPI service that searches
that index and returns the most relevant passages.

## Progress log
 
- **Step 1**: Extract text from each PDF page and print a char count per page,
  to confirm PDF text extraction works before building anything on top of it.
  Log a warning when a PDF yields 0 chars across all pages, without attempting OCR — documented as a known limitation below.
- **Step 2**: Split each page's text into overlapping character-based chunks and print a preview of each, to confirm chunk boundaries look reasonable before generating embeddings.

- **Step 3**: Load the multilingual embedding model and embed all chunks in one batch, printing the resulting shape, to confirm the model runs on CPU and produces vectors of the expected dimension before building the index.

 ## Review

 #### Scanned/image-only PDFs.
 No text layer means pdfplumber extracts 0 chars, so the file can't be searched. Detected and logged as a warning; OCR was out of scope for this exercise.

 #### Tables get flattened into meaningless text.
 Tables get flattened into meaningless text. One PDF has a table (N° Acte | Vendeur | Date | Cadastre | Surface). extract_text() reads left-to-right by position, not by column, so it mixes a cadastral reference, a name fragment, and a date into one line with no structure. Search over these chunks won't reliably match a query about a specific row. A real fix would use page.extract_tables() to parse rows properly — left out here to stay within scope.