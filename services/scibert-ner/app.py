"""SciBERT NER Microservice."""

import sys, os
sys.path.insert(0, "/app")

from fastapi import FastAPI
from pydantic import BaseModel
from materials_ai.modules.nlp_literature_mining.scibert_ner import SciBERTNER

app = FastAPI(title="SciBERT NER Service", version="1.0.0")
ner = SciBERTNER()


class NERRequest(BaseModel):
    text: str


class NERResponse(BaseModel):
    entities: list


@app.post("/predict")
async def predict(body: NERRequest):
    entities = ner.extract_entities(body.text)
    return NERResponse(entities=[e.to_dict() for e in entities])


@app.get("/health")
async def health():
    return {"status": "healthy"}
