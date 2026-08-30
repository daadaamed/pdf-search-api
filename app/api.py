from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from app.config import DATA_DIR
from app.indexing import load_index, embed_and_search

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        index, metadata, model_name = load_index(DATA_DIR)
        print(f"Loading embedding model ({model_name})...")
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Could not load index from '{DATA_DIR}/' ({e}). "
            f"Have you run ingestion yet? Try: python -m app.ingest <path_to_pdf_folder>"
        ) from e

    state["index"] = index
    state["metadata"] = metadata
    state["model"] = model

    print(f"Ready: {index.ntotal} vectors loaded.")
    yield
    state.clear()


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
    return {
        "status": "ok",
        "vectors_loaded": state["index"].ntotal,
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    results = embed_and_search(state["model"], state["index"], state["metadata"], request.query, request.top_k)
    return SearchResponse(query=request.query, results=[SearchResult(**r) for r in results])