"""Module 1: 文献挖掘 API."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from fastapi import APIRouter, Depends, HTTPException
from ...core.dependencies import get_current_user
from ...schemas.common import LiteratureQuery, LiteratureAnalysis, RAGQuery, APIResponse

router = APIRouter(prefix="/literature", tags=["文献挖掘"])


@router.post("/search")
async def search_papers(body: LiteratureQuery, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.nlp_literature_mining.downloader import PaperDownloader
        dl = PaperDownloader()
        if body.source == "arxiv":
            papers = dl.search_arxiv(body.query, max_results=body.max_results)
        else:
            papers = dl.search_semantic_scholar(body.query, max_results=body.max_results)
        return APIResponse(success=True, data={"papers": papers, "count": len(papers)})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/analyze")
async def analyze_text(body: LiteratureAnalysis, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.nlp_literature_mining.scibert_ner import SciBERTNER
        ner = SciBERTNER()
        entities = ner.extract_entities(body.text)
        return APIResponse(success=True, data={"entities": [e.to_dict() for e in entities]})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/extract-triplets")
async def extract_triplets(body: LiteratureAnalysis, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.nlp_literature_mining.scibert_ner import SciBERTNER
        from materials_ai.modules.nlp_literature_mining.relation_extractor import RelationExtractor
        ner = SciBERTNER()
        entities = ner.extract_entities(body.text)
        extractor = RelationExtractor()
        triplets = extractor.extract_triplets(body.text, entities)
        return APIResponse(success=True, data={"triplets": [t.__dict__ for t in triplets], "count": len(triplets)})
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/rag")
async def rag_query(body: RAGQuery, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.nlp_literature_mining.rag_pipeline import MaterialsRAG
        rag = MaterialsRAG()
        if not rag.is_ready:
            return APIResponse(success=False, message="RAG索引未建立, 请先导入论文")
        result = rag.ask(body.question, k=body.k)
        return APIResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(500, str(e))

