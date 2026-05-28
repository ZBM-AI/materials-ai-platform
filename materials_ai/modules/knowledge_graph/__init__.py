"""材料科学知识图谱模块 v2 — Neo4j + RGCN + 本体驱动"""

from .ontology import (
    NODE_LABELS, RELATION_TYPES, NODE_PROPERTIES,
    ENTITY_COLORS, ENTITY_LABELS_ZH, RELATION_LABELS_ZH, CYPHER_TEMPLATES,
)
from .graph_builder import KnowledgeGraphBuilder, HAS_NEO4J
from .graph_query import GraphQuery
from .graph_viz import GraphVisualizer
from .graph_store import GraphStore
from .gnn_link_predictor import (
    MaterialKnowledgeGraph, RGCNLinkPredictor, train_link_prediction,
    HAS_TORCH, HAS_PYG,
)
