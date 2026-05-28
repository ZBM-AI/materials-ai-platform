"""Embedding API Microservice — 文本向量化."""

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Embedding Service", version="1.0.0")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class EmbedRequest(BaseModel):
    texts: list[str]
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int


@app.post("/embed")
async def embed(body: EmbedRequest):
    embeddings = model.encode(
        body.texts, normalize_embeddings=body.normalize, show_progress_bar=False
    )
    return EmbedResponse(
        embeddings=embeddings.tolist(),
        dimension=embeddings.shape[1],
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "all-MiniLM-L6-v2"}
