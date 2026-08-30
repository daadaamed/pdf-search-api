import json

import faiss


def build_index(model, metadata):
    """Embed metadata's chunks and build a normalized FAISS IndexFlatIP.

    `model` is a caller-provided SentenceTransformer instance — never
    instantiated here, so the same loaded model can be reused for both
    ingestion and, separately, query-time search without doubling memory.

    `metadata` is the full list of chunk-metadata dicts (each already
    carrying its own "text"), not a separately-threaded list of raw
    strings — this is also what gets persisted alongside the index.
    """
    texts = [m["text"] for m in metadata]
    embeddings = model.encode(texts, show_progress_bar=True).astype("float32")
    assert embeddings.shape[0] == len(metadata), "embeddings and metadata are out of sync!"

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index, metadata, model_name, output_dir):
    """Persist the index, metadata, and model info to output_dir."""
    output_dir.mkdir(exist_ok=True)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(output_dir / "index_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {"model_name": model_name, "dimension": index.d, "num_vectors": index.ntotal},
            f, ensure_ascii=False, indent=2,
        )


def load_index(data_dir):
    """Load the index, metadata, and model name from data_dir.

    Raises FileNotFoundError/RuntimeError (from missing files or FAISS's
    own read errors) or ValueError (on an index/metadata count mismatch)
    without any web-framework-specific wrapping — callers like the API's
    startup are responsible for translating these into user-facing
    messages.
    """
    with open(data_dir / "index_info.json", encoding="utf-8") as f:
        info = json.load(f)

    index = faiss.read_index(str(data_dir / "index.faiss"))
    with open(data_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"index has {index.ntotal} vectors but metadata.json has "
            f"{len(metadata)} entries — they're out of sync"
        )

    return index, metadata, info["model_name"]


def embed_and_search(model, index, metadata, query, top_k):
    """Embed query with model, search index, and return matched metadata + score."""
    query_vector = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS pads with -1 if top_k > number of vectors in the index
            continue
        meta = metadata[idx]
        results.append({
            "document_name": meta["document_name"],
            "page_number": meta["page_number"],
            "chunk_index": meta["chunk_index"],
            "score": float(score),
            "text": meta["text"],
        })
    return results