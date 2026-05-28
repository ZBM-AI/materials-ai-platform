"""知识图谱查询 v2 — NetworkX查询 + Neo4j Cypher查询"""

from typing import List, Optional, Dict, Any
from difflib import get_close_matches
import re

import networkx as nx

from .ontology import CYPHER_TEMPLATES, ENTITY_LABELS_ZH, RELATION_LABELS_ZH


class GraphQuery:
    """材料知识图谱查询引擎.

    双后端:
    - NetworkX: 本地内存查询 (始终可用)
    - Neo4j: Cypher查询 (用于复杂图查询和推理)
    """

    def __init__(self, graph: nx.MultiDiGraph, neo4j_graph=None):
        self.graph = graph
        self._neo4j = neo4j_graph

    # ================================================================
    # NetworkX 查询 (向后兼容)
    # ================================================================

    def get_entity(self, name: str) -> Optional[dict]:
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("name", "").lower() == name.lower():
                return {"id": node_id, **attrs}
        return None

    def get_neighbors(self, name: str) -> dict:
        entity = self.get_entity(name)
        if not entity:
            return {"entity": None, "relations": []}
        node_id = entity["id"]
        relations = []
        for _, neighbor, attrs in self.graph.out_edges(node_id, data=True):
            nd = dict(self.graph.nodes[neighbor])
            relations.append({
                "direction": "outgoing",
                "predicate": attrs.get("predicate", ""),
                "target": {"id": neighbor, **nd},
                "evidence": attrs.get("evidence", ""),
            })
        for neighbor, _, attrs in self.graph.in_edges(node_id, data=True):
            nd = dict(self.graph.nodes[neighbor])
            relations.append({
                "direction": "incoming",
                "predicate": attrs.get("predicate", ""),
                "target": {"id": neighbor, **nd},
                "evidence": attrs.get("evidence", ""),
            })
        return {"entity": entity, "relations": relations}

    def search(self, keyword: str, limit: int = 50) -> List[dict]:
        kw = keyword.lower()
        results = []
        for node_id, attrs in self.graph.nodes(data=True):
            if kw in attrs.get("name", "").lower():
                results.append({"id": node_id, **attrs})
                if len(results) >= limit:
                    break
        return results

    def fuzzy_search(self, keyword: str, limit: int = 20) -> List[dict]:
        all_names = [attrs.get("name", "") for _, attrs in self.graph.nodes(data=True)]
        matched = get_close_matches(keyword, all_names, n=limit, cutoff=0.6)
        results = []
        for name in matched:
            for node_id, attrs in self.graph.nodes(data=True):
                if attrs.get("name", "") == name:
                    results.append({"id": node_id, **attrs})
                    break
        return results

    def find_by_type(self, entity_type: str) -> List[dict]:
        return [
            {"id": nid, **attrs}
            for nid, attrs in self.graph.nodes(data=True)
            if attrs.get("entity_type") == entity_type
        ]

    def find_path(self, name1: str, name2: str, max_length: int = 4) -> Optional[List[dict]]:
        e1 = self.get_entity(name1)
        e2 = self.get_entity(name2)
        if not e1 or not e2:
            return None
        try:
            path = nx.shortest_path(
                self.graph.to_undirected(),
                source=e1["id"], target=e2["id"],
            )
            if len(path) - 1 > max_length:
                return None
            return [
                {"step": i, "id": nid, **dict(self.graph.nodes[nid])}
                for i, nid in enumerate(path)
            ]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_subgraph(self, entity_names: List[str], depth: int = 1) -> nx.MultiDiGraph:
        node_ids = set()
        for name in entity_names:
            e = self.get_entity(name)
            if e:
                node_ids.add(e["id"])
        if not node_ids:
            return nx.MultiDiGraph()
        subgraph_nodes = set(node_ids)
        for _ in range(depth):
            new_nodes = set()
            for nid in list(subgraph_nodes):
                new_nodes.update(self.graph.successors(nid))
                new_nodes.update(self.graph.predecessors(nid))
            subgraph_nodes.update(new_nodes)
        return self.graph.subgraph(subgraph_nodes).copy()

    def get_statistics(self) -> dict:
        entity_counts = {}
        for _, attrs in self.graph.nodes(data=True):
            t = attrs.get("entity_type", "unknown")
            entity_counts[t] = entity_counts.get(t, 0) + 1
        relation_counts = {}
        for _, _, attrs in self.graph.edges(data=True):
            p = attrs.get("predicate", "unknown")
            relation_counts[p] = relation_counts.get(p, 0) + 1
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "entity_types": entity_counts,
            "relation_types": relation_counts,
        }

    # ================================================================
    # Neo4j Cypher 查询
    # ================================================================

    def cypher_query(self, query: str, params: dict = None) -> List[dict]:
        """在Neo4j中执行原始Cypher查询."""
        if self._neo4j is None:
            raise RuntimeError("Neo4j not available")
        params = params or {}
        cursor = self._neo4j.run(query, params)
        return [dict(record) for record in cursor]

    def find_improve_strength_ductility(self, material_name: str = "steel") -> List[dict]:
        """查找可提高某材料屈服强度且不显著降低延伸率的工艺."""
        if self._neo4j is None:
            return self._nx_find_improve_process(material_name)
        return self.cypher_query(
            CYPHER_TEMPLATES["improve_strength_ductility"],
            {"material_name": material_name},
        )

    def recommend_materials(self, property_name: str, min_processes: int = 2,
                            limit: int = 10) -> List[dict]:
        """推荐潜在高性能材料."""
        if self._neo4j is None:
            return []
        return self.cypher_query(
            CYPHER_TEMPLATES["recommend_high_performance"],
            {"property_name": property_name, "min_processes": min_processes, "limit": limit},
        )

    def query_process_effects(self, process_name: str) -> List[dict]:
        """查询某工艺影响的所有性能和微观结构."""
        if self._neo4j is None:
            return []
        return self.cypher_query(
            CYPHER_TEMPLATES["process_effects"],
            {"process_name": process_name},
        )

    def compare_materials(self, material1: str, material2: str) -> List[dict]:
        """对比两种材料的属性、工艺、结构、应用."""
        if self._neo4j is None:
            return self._nx_compare_materials(material1, material2)
        return self.cypher_query(
            CYPHER_TEMPLATES["compare_materials"],
            {"material1": material1, "material2": material2},
        )

    def export_subgraph_cypher(self, keyword: str) -> List[dict]:
        """导出包含某关键词的子图 (节点+关系)."""
        if self._neo4j is None:
            return []
        return self.cypher_query(
            CYPHER_TEMPLATES["export_subgraph"],
            {"keyword": keyword},
        )

    def get_material_property_pairs(self) -> List[dict]:
        """获取所有 (材料, 性能) 对 — 用于RGCN训练."""
        if self._neo4j is None:
            return self._nx_material_property_pairs()
        return self.cypher_query(CYPHER_TEMPLATES["all_material_property_pairs"])

    def find_microstructure_mediated(self, microstructure: str,
                                      property_name: str, limit: int = 20) -> List[dict]:
        """查找通过特定微观结构提升性能的工艺."""
        if self._neo4j is None:
            return []
        return self.cypher_query(
            CYPHER_TEMPLATES["microstructure_mediated_improvement"],
            {"microstructure": microstructure, "property": property_name, "limit": limit},
        )

    # ================================================================
    # NetworkX 回退实现
    # ================================================================

    def _nx_find_improve_process(self, material_name: str) -> List[dict]:
        """NetworkX版本的工艺推荐 (简化)."""
        results = []
        kw = material_name.lower()
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("entity_type") not in ("material", "Material"):
                continue
            if kw not in attrs.get("name", "").lower():
                continue
            for _, neighbor, edge in self.graph.out_edges(node_id, data=True):
                if edge.get("predicate") in ("processedBy", "usesProcess"):
                    proc = dict(self.graph.nodes[neighbor])
                    results.append({
                        "process": proc.get("name", ""),
                        "material": attrs.get("name", ""),
                    })
        return results

    def _nx_compare_materials(self, m1: str, m2: str) -> List[dict]:
        n1 = self.get_neighbors(m1)
        n2 = self.get_neighbors(m2)
        return [{
            "material1": m1,
            "relations1": [r["predicate"] + "→" + r["target"]["name"] for r in n1["relations"]],
            "material2": m2,
            "relations2": [r["predicate"] + "→" + r["target"]["name"] for r in n2["relations"]],
        }]

    def _nx_material_property_pairs(self) -> List[dict]:
        pairs = []
        mat_nodes = {nid for nid, a in self.graph.nodes(data=True)
                     if a.get("entity_type") in ("material", "Material")}
        prop_nodes = {nid for nid, a in self.graph.nodes(data=True)
                      if a.get("entity_type") in ("property", "Property")}

        existing = set()
        for u, v, _ in self.graph.edges(data=True):
            if u in mat_nodes and v in prop_nodes:
                existing.add((u, v))

        for m_id in mat_nodes:
            for p_id in prop_nodes:
                pairs.append({
                    "material": self.graph.nodes[m_id].get("name", m_id),
                    "property": self.graph.nodes[p_id].get("name", p_id),
                    "exists": 1 if (m_id, p_id) in existing else 0,
                })
        return pairs
