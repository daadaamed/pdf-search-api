from fastapi import FastAPI

app = FastAPI(title="PDF Search API")


@app.get("/health")
def health():
    return {"status": "ok"}