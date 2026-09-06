# PDF Search API
 
A small local system to search across a collection of PDF documents by meaning
(semantic search), not just keywords. Built as a CLI ingestion program (PDFs →
text → chunks → embeddings → FAISS index) plus a FastAPI service that searches
that index and returns the most relevant passages.

## Usage

### Ingestion
 
```bash
pip install -r requirements.txt
python -m app.ingest <path_to_pdf_folder>
```
 
Run as a module (`-m app.ingest`), not as a direct script path (`python app/ingest.py`); the code lives inside the `app` package and uses absolute imports.

This produces three files in `data/`.
Re-run the same command any time the PDF folder changes; ingestion rebuilds from scratch.

### API server
 
```bash
uvicorn app.api:app --reload
```

Check `http://127.0.0.1:8000/health` — should return `{"status": "ok"}`.
Interactive docs (Swagger UI) are auto-generated at `http://127.0.0.1:8000/docs`.

Example search request:
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Quelle est la position du document sur les politiques publiques ?", "top_k": 5}'
```

Example response:
```json
{
  "query": "Quelle est la position du document sur les politiques publiques ?",
  "results": [
    {
      "document_name": "example.pdf",
      "page_number": 3,
      "chunk_index": 12,
      "score": 0.82,
      "text": "Contenu du passage correspondant..."
    }
  ]
}
```

### Docker

Build the image:
```bash
docker build -t pdf-search .
```

Run ingestion (mounts your local PDF folder and a `data/` folder so the generated index persists on your host machine after the container exits):
```bash
docker run --rm \
  -v $(pwd)/pdfs:/app/pdfs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  pdf-search \
  python -m app.ingest /app/pdfs
```

Verify it worked by checking your host's `data/` folder; you should see `index.faiss`, `metadata.json`, and `index_info.json`. Start the API (same `data/` volume, so it sees what ingestion produced):
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  -p 8000:8000 \
  pdf-search
```

Then call `/search` exactly as shown above, against `http://127.0.0.1:8000`. To rebuild the index after changing the PDF folder, just re-run the ingestion command; it always rebuilds `data/` from scratch.

Example request:
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "montant du don pour le mécénat", "top_k": 3}'
```

 ## Review

  ### Assumptions
 
- The PDF folder is trusted, local input.
- There's no file-watching, no reload endpoint, nothing that re-reads data/ after boot.
- Ingestion is a full rebuild every run, not incremental.
- One chunk size/overlap for every document, regardless of content.

### Functional

- chunk_index resets per document, not per page.
- One corrupt/unreadable PDF is skipped, not fatal to the whole ingestion run.
- Model name and dimension are persisted in index_info.json so ingestion and the API can never silently drift onto different embedding spaces.
- API startup validates data/ fully before serving traffic: missing files, corrupt files, or an index/metadata count mismatch raise a clear error.
- Code is organized as an app/ package (config, pdf_extraction, chunking, indexing, ingest, api); one concern per module, with indexing.py shared by both entrypoints so FAISS logic isn't duplicated.
- Both docker run commands mount hf_cache/ to /root/.cache/huggingface, so the embedding model downloads once and is reused across the separate ingestion and API containers — without it, each --rm container's cache is discarded on exit and the ~470MB model re-downloads every run.

#### Sentence transformers
- Chosen: `paraphrase-multilingual-MiniLM-L12-v2`: small, fast, CPU-friendly, adequate multilingual quality for this exercise's scale.
- If prioritizing retrieval quality over speed: `paraphrase-multilingual-mpnet-base-v2` (same family, ~2x cost, meaningfully better accuracy).
- If the corpus were guaranteed French-only: a French-specific model (e.g. CamemBERT-based sentence embeddings) could outperform multilingual options, at the cost of losing support for other languages.

 ### Limitation

- Text extraction is layout-blind (`pdfplumber` reads by position, not structure); the root cause behind most search-quality issues below.
- Chunking is character-based, not sentence-aware (overlap mitigates but doesn't eliminate mid-sentence cuts). A separate branch (feat/add-paragraphs-splitter-tokenized) implements paragraph → sentence → word/token-aware splitting as an improvement on this.
- No keyword search, nor re-ranking. So, if someone searches for "AFF-2026.06.11-DP-26A0019", dense search might not surface it reliably. => It's comparing meaning-fingerprints, not text.


### Situations where search quality may be poor
 
- Scanned/image-only PDFs (no text layer to search).
- Tables (flattened text loses row/column structure — real example found
  in the source PDFs).
- Very short chunks (e.g. a lone page number) are indexed and add minor noise.

## What I'd improve for a production-grade, scalable system (depends on constraints, data type, SLOs...)

- **Ingestion pipeline**: make it incremental (add/update/delete a single document without a full rebuild) and run it as an async background job instead of a blocking CLI script, for when documents are added or changed regularly (depending on business requirements). Add basic input guards (max file size, max page count, per-file timeout) so one huge or malformed PDF can't stall or crash a run.
- **Chunk identity & lifecycle**: give each chunk a stable `chunk_id` that survives re-ingestion, so upserts/deletes can target it directly; version chunks when the chunking strategy changes, so old and new chunks aren't silently mixed in one index; add access control if documents ever belong to different users/tenants, since nothing today scopes a search to a subset of the corpus.
- **Chunking strategy**: detect content type per page/section (table vs. prose vs. form) and route to a matching strategy, table rows via `pdfplumber.extract_tables()` (headers repeated per row), recursive character splitting for prose (replacing today's fixed-size window). Attach context (section heading, table caption) to each chunk at creation time, not as a later pass. Re-evaluate semantic chunking only if prose retrieval quality still falls short after this.
- **OCR**: route scanned/image-only PDFs instead of skipping them.
- **Vector store - latency**: swap `IndexFlatIP` for an approximate index (HNSW/IVF) if raw search latency becomes the bottleneck at larger scale.
- **Vector store - mutability**: move to a managed/self-hosted vector DB (Qdrant, pgvector) if the real need is per-document upsert/delete or metadata filtering at the storage layer.
- **Search quality**: add hybrid search and a re-ranking pass; pure embedding similarity misses exact terms like names and reference numbers.
- **Backup / rollback**: since ingestion always rebuilds `data/` from scratch, a bad run (bug, corrupted PDF, wrong model) silently replaces a working index with a broken one. Snapshot `data/` before each rebuild (even a simple copy to `data.bak/`) so recovery is restoring a snapshot, not re-running ingestion from the original PDFs.
- **API**: add auth, rate limiting, response caching for repeated/common queries, and horizontal scaling depending on business requirements.
- **Observability**: structured logging and metrics instead of `print()` statements; e.g. query latency, 0-result searches, index size over time.
- **Model lifecycle**: a documented way to re-embed and hot-swap the index when the embedding model changes.
- **Testing**: unit tests (API), integration tests, end-to-end tests.


## Progress log (iterations)
 
- **Step 1**: Extract text from each PDF page and print a char count per page, to confirm PDF text extraction works before building anything on top of it.
- **Step 2**: Split each page's text into overlapping character-based chunks and print a preview of each, to confirm chunk boundaries look reasonable before generating embeddings.

- **Step 3**: Load the multilingual embedding model and embed all chunks in one batch, printing the resulting shape, to confirm the model runs on CPU and produces vectors of the expected dimension before building the index.

- **Step 4**: Normalize embeddings, build a FAISS `IndexFlatIP` index (cosine similarity via inner product), and persist the index, metadata, and an `index_info.json` (model name + dimension) to `data/`, so the API can verify it's using the same embedding model. Re-running ingestion replaces `data/` entirely.

- **Step 5**: Add accepting the folder path as a CLI argument using `sys.argv`

- **Step 6**: Minimal FastAPi app with /health reporting the number of vectors loaded.
Load the FAISS index, metadata, and embedding model once at startup (via FastAPI's lifespan), reading the model name from index_info.json rather than a hardcoded constant so ingestion and the API can't silently drift apart. 

- **Step 7**: Implement POST /search — embed the query with the same model and normalization as ingestion, search the FAISS index, and map results back to their metadata. This is the core deliverable of the API.

- **Step 8**: Dockerize the app.

- **Step 9**: Refactor the app, using better folder structure and reusable modules.

- **Step 10**: Refactored from two flat scripts into an `app/` package