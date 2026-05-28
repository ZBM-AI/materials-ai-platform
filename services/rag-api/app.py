"""RAG 问答 API Microservice."""

import sys, os
sys.path.insert(0, "/app")

from fastapi import FastAPI
from pydantic import BaseModel
from materials_ai.modules.nlp_literature_mining.rag_pipeline import MaterialsRAG

app = FastAPI(title="RAG API Service", version="1.0.0")
rag = MaterialsRAG()


class RAGRequest(BaseModel):
    question: str
    k: int = 5


class RAGResponse(BaseModel):
    answer: str
    sources: list


@app.post("/ask")
async def ask(body: RAGRequest):
    result = rag.ask(body.question, k=body.k)
    return RAGResponse(answer=result.get("answer", ""), sources=result.get("sources", []))


@app.get("/health")
async def health():
    return {"status": "healthy", "indexed": rag.is_ready}
