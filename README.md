# PDF Search API
 
A small local system to search across a collection of PDF documents by meaning
(semantic search), not just keywords. Built as a CLI ingestion program (PDFs →
text → chunks → embeddings → FAISS index) plus a FastAPI service that searches
that index and returns the most relevant passages.
 
## Progress log
 
- **Step 1**: Extract text from each PDF page and print a char count per page,
  to confirm PDF text extraction works before building anything on top of it.
 