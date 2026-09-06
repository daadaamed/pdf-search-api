from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import logging

from app.config import DATA_DIR
from app.indexing import load_index, embed_and_search

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    try:
        index, metadata, model_name = load_index(DATA_DIR)
        logger.info(f"Loading embedding model ({model_name})...")
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Could not load index from '{DATA_DIR}/' ({e}). "
            f"Have you run ingestion yet? Try: python -m app.ingest <path_to_pdf_folder>"
        ) from e

    app.state.index = index
    app.state.metadata = metadata
    app.state.model = model
    app.state.ready = True

    logger.info(f"Ready: {index.ntotal} vectors loaded.")
    yield
    app.state.ready = False


app = FastAPI(title="PDF Search API", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0, le=50)


class SearchResult(BaseModel):
    document_name: str
    page_number: int
    chunk_index: int
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


@app.get("/health")
def health():
    if not app.state.ready:
        return {"status": "starting"}
    return {"status": "ok", "vectors_loaded": app.state.index.ntotal}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        results = embed_and_search(
            app.state.model, app.state.index, app.state.metadata,
            request.query, request.top_k
        )
    except Exception:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail="Search failed")

    return SearchResponse(query=request.query, results=[SearchResult(**r) for r in results])