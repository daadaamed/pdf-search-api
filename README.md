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
Re-run the same command any time the PDF folder changes; ingestion always rebuilds from scratch rather than appending or updating in place.

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

Note: the first run downloads ~470MB of embedding model weights from Hugging Face and needs internet access to do so.

Verify it worked by checking your host's `data/` folder; you should see `index.faiss`, `metadata.json`, and `index_info.json`.Start the API (same `data/` volume, so it sees what ingestion produced):
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
  -p 8000:8000 \
  pdf-search
```

Then call `/search` exactly as shown above, against `http://127.0.0.1:8000`.To rebuild the index after changing the PDF folder, just re-run the ingestion command; it always rebuilds `data/` from scratch.


Example  request:
```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "montant du don pour le mécénat", "top_k": 3}'
```

Example response:
```json
{
  "query": "montant du don pour le mécénat",
  "results": [
    {
      "document_name": "159687_Projet+de+convention+de+mecenat+financier_BOYER+GESTION-2026+04+28+1.pdf",
      "page_number": 6,
      "chunk_index": 26,
      "score": 0.713,
      "text": "ANNEXES\nANNEXE 1 - GRILLE DE CONTREPARTIES\nMontant du don (par Contreparties\ntranches)..."
    },
    {
      "document_name": "159687_Projet+de+convention+de+mecenat+financier_BOYER+GESTION-2026+04+28+1.pdf",
      "page_number": 2,
      "chunk_index": 5,
      "score": 0.638,
      "text": "4-1 Montant du don\nLe MÉCÈNE s'engage à soutenir le projet par un don financier à hauteur de :\n(somme en lettres euros) Trois mille cinq cent euros\n(somme en chiffre) 3 500,00 €..."
    }
  ]
}
```

## Progress log
 
- **Step 1**: Extract text from each PDF page and print a char count per page,
  to confirm PDF text extraction works before building anything on top of it.
  Log a warning when a PDF yields 0 chars across all pages, without attempting OCR: documented as a known limitation below.
- **Step 2**: Split each page's text into overlapping character-based chunks and print a preview of each, to confirm chunk boundaries look reasonable before generating embeddings.
  Embeddings and metadata are kept aligned by construction, not by lookup.

- **Step 3**: Load the multilingual embedding model and embed all chunks in one batch, printing the resulting shape, to confirm the model runs on CPU and produces vectors of the expected dimension before building the index.

- **Step 4**: Normalize embeddings, build a FAISS `IndexFlatIP` index (cosine similarity via inner product), and persist the index, metadata, and an `index_info.json` (model name + dimension) to `data/`, so the API can verify it's using the same embedding model. Re-running ingestion replaces `data/` entirely.

- **Step 5**: Add accepting the folder path as a CLI argument using `sys.argv`

- **Step 6**: Minimal FastAPi app with /health reporting the number of vectors loaded.
Load the FAISS index, metadata, and embedding model once at startup (via FastAPI's lifespan), reading the model name from index_info.json rather than a hardcoded constant so ingestion and the API can't silently drift apart. 

- **Step 7**: Implement POST /search — embed the query with the same model and normalization as ingestion, search the FAISS index, and map results back to their metadata. This is the core deliverable of the API.

- **Step 8**: Dockerize the app.

- **Step 9**: Refactor the app, using better folder structure and reusable modules.

- **Step 10**: Refactored from two flat scripts into an `app/` package

 ## Review

- chunk_index resets per document, not per page.
- One corrupt/unreadable PDF is skipped, not fatal to the whole ingestion run.
- Model name and dimension are persisted in index_info.json so ingestion and the API can never silently drift onto different embedding spaces.
- API startup validates data/ fully before serving traffic: missing files, corrupt files, or an index/metadata count mismatch all raise a clear error immediately, rather than serving silently wrong results.
- Code is organized as an app/ package (config, pdf_extraction, chunking, indexing, ingest, api); one concern per module, with indexing.py shared by both entrypoints so FAISS logic isn't duplicated.
- FAISS was chosen over higher-level options like ChromaDB to keep the indexing/normalization logic explicit and inspectable.
- Both docker run commands mount hf_cache/ to /root/.cache/huggingface, so the embedding model downloads once and is reused across the separate ingestion and API containers — without it, each --rm container's cache is discarded on exit and the ~470MB model re-downloads every run.

#### Sentence transformers
- Chosen: `paraphrase-multilingual-MiniLM-L12-v2`: small, fast, CPU-friendly, adequate multilingual quality for this exercise's scale.
- If prioritizing retrieval quality over speed: `paraphrase-multilingual-mpnet-base-v2` (same family, ~2x cost, meaningfully better accuracy).
- If the corpus were guaranteed French-only: a French-specific model (e.g. CamemBERT-based sentence embeddings) could outperform multilingual options, at the cost of losing support for other languages.

 ### Limitation

- Text extraction is layout-blind (`pdfplumber` reads by position, not structure) — the root cause behind most search-quality issues below.
- Chunking is character-based, not sentence-aware (overlap mitigates but doesn't eliminate mid-sentence cuts).
- No keyword search, nor re-ranking. So, if someone searches for "AFF-2026.06.11-DP-26A0019", dense search might not surface it reliably. => It's comparing meaning-fingerprints, not text.

 ### Assumptions
 
- The PDF folder is trusted, local input.
- Ingestion is a full rebuild every run, not incremental.
- One embedding model per index, fixed at ingestion time.


### Situations where search quality may be poor
 
- Scanned/image-only PDFs (no text layer to search).
- Tables (flattened text loses row/column structure — real example found
  in the source PDFs).
- Very short chunks (e.g. a lone page number) are indexed and add minor noise.

## What I'd improve for a production-grade, scalable system (depends with constraints, type of data, slo...)

- **Vector store**: swap `IndexFlatIP` for an approximate index (HNSW/IVF) if raw search latency becomes the bottleneck separately, move to a managed/self-hosted vector DB (Qdrant, pgvector) if the actual need is incremental upsert/delete or metadata filtering, which a flat FAISS file can't do.
- **Ingestion**: make it incremental (add/update/delete single documents) and run as an async background job, not a blocking CLI script in case documents start getting added or changed regularly.
- **Chunking**: move to document-structure chunking first, scoped specifically to the tabular PDFs where the current chunker is already known to scramble content (split by table row using `pdfplumber.extract_tables()`, headers repeated per row); apply recursive character splitting as the default for plain prose, replacing today's fixed-size window. Route each page/section to the method matching its actual content type (table vs. prose vs. form fields) rather than applying one strategy per whole document, since a single PDF often mixes several. Re-evaluate with semantic chunking only if measured retrieval quality on non-tabular content still falls short after this change.
- **Search quality**: add hybrid search and a re-ranking pass; pure embedding similarity misses exact terms like names and reference numbers.
- **OCR**: route scanned/image-only PDFs through Tesseract instead of skipping them.
- **API**: add auth, rate limiting, caching, horizontal scaling...
- **Observability**: structured logging and metrics instead of `print()` statements. E.g: log query latency, log when a search returns 0 results, track index size over time.
- **Model lifecycle**: a documented way to re-embed and hot-swap the index when the embedding model changes
- **Add Test scenarios**: add unit tests(api), integrations test, end-to-end tests..