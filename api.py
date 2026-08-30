from contextlib import asynccontextmanager
from pathlib import Path
import json

import faiss
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with open(DATA_DIR / "index_info.json", encoding="utf-8") as f:
            info = json.load(f)

        print(f"Loading embedding model ({info['model_name']})...")
        state["model"] = SentenceTransformer(info["model_name"])

        print("Loading FAISS index and metadata...")
        state["index"] = faiss.read_index(str(DATA_DIR / "index.faiss"))
        with open(DATA_DIR / "metadata.json", encoding="utf-8") as f:
            state["metadata"] = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Could not load index from '{DATA_DIR}/' ({e}). "
            f"Have you run ingestion yet? Try: python ingest.py <path_to_pdf_folder>"
        ) from e

    if state["index"].ntotal != len(state["metadata"]):
        raise RuntimeError(
            f"Index has {state['index'].ntotal} vectors but metadata.json has "
            f"{len(state['metadata'])} entries — they're out of sync. "
            f"Re-run ingestion: python ingest.py <path_to_pdf_folder>"
        )

    print(f"Ready: {state['index'].ntotal} vectors loaded.")
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
    query_vector = state["model"].encode([request.query]).astype("float32")
    faiss.normalize_L2(query_vector)

    scores, indices = state["index"].search(query_vector, request.top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = state["metadata"][idx]
        results.append(SearchResult(
            document_name=meta["document_name"],
            page_number=meta["page_number"],
            chunk_index=meta["chunk_index"],
            score=float(score),
            text=meta["text"],
        ))

    return SearchResponse(query=request.query, results=results)