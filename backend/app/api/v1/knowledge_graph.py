"""Module 2: 知识图谱 API."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from fastapi import APIRouter, HTTPException, Depends
from ...core.dependencies import get_current_user
from ...schemas.common import KGQuery, APIResponse

router = APIRouter(prefix="/kg", tags=["知识图谱"])


@router.post("/query")
async def graph_query(body: KGQuery, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.knowledge_graph.graph_query import GraphQuery
        from utils.data_loader import DataLoader
        seed_kg = DataLoader.load_seed_kg()
        gq = GraphQuery(seed_kg)
        if body.entity_name:
            nodes = gq.find_entity(body.entity_name)
            return APIResponse(success=True, data={"nodes": nodes})
        if body.cypher:
            results = gq.execute_cypher(body.cypher)
            return APIResponse(success=True, data={"results": results})
        return APIResponse(success=False, message="请提供 entity_name 或 cypher")
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/stats")
async def graph_stats(user=Depends(get_current_user)):
    try:
        from materials_ai.modules.knowledge_graph.graph_builder import KnowledgeGraphBuilder
        from utils.data_loader import DataLoader
        builder = KnowledgeGraphBuilder()
        seed_kg = DataLoader.load_seed_kg()
        graph = builder.build_from_extractions([], seed_kg)
        return APIResponse(success=True, data={
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/recommend/{property_name}")
async def recommend_materials(property_name: str, user=Depends(get_current_user)):
    try:
        from materials_ai.modules.knowledge_graph.graph_query import GraphQuery
        from utils.data_loader import DataLoader
        seed_kg = DataLoader.load_seed_kg()
        gq = GraphQuery(seed_kg)
        recs = gq.recommend_materials(property_name)
        return APIResponse(success=True, data={"recommendations": recs})
    except Exception as e:
        raise HTTPException(500, str(e))
