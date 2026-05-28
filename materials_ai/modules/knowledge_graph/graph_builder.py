"""知识图谱构建器 v2 — NetworkX + Neo4j (py2neo) 双后端"""

import hashlib
import os
from typing import List, Optional, Dict

import networkx as nx

from .ontology import (
    NODE_LABELS, RELATION_TYPES, NODE_PROPERTIES,
    ENTITY_LABELS_ZH, RELATION_LABELS_ZH, ENTITY_COLORS,
)

# ---- Neo4j (optional) ----
try:
    from py2neo import Graph as Neo4jGraph, Node, Relationship, Subgraph
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    Neo4jGraph = None  # type placeholder when py2neo not installed


class KnowledgeGraphBuilder:
    """材料科学知识图谱构建器.

    支持双后端:
    - NetworkX (始终可用, 用作内存分析和可视化)
    - Neo4j (通过py2neo, 用于持久化存储和Cypher查询)
    """

    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None,
                 neo4j_password: str = None):
        from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        self.neo4j_uri = neo4j_uri or NEO4J_URI
        self.neo4j_user = neo4j_user or NEO4J_USER
        self.neo4j_password = neo4j_password or NEO4J_PASSWORD
        self.graph = nx.MultiDiGraph()
        self._entity_map: Dict[str, str] = {}
        self._neo4j_graph = None

    # ================================================================
    # Neo4j 连接
    # ================================================================

    @property
    def neo4j(self):
        """懒加载Neo4j连接."""
        if self._neo4j_graph is None and HAS_NEO4J:
            try:
                self._neo4j_graph = Neo4jGraph(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                )
                self._neo4j_graph.run("RETURN 1")
            except Exception as e:
                print(f"  [Neo4j connection failed] {e}")
                self._neo4j_graph = None
        return self._neo4j_graph

    @property
    def neo4j_available(self) -> bool:
        return self.neo4j is not None

    # ================================================================
    # Schema 初始化 (Neo4j约束和索引)
    # ================================================================

    def init_neo4j_schema(self):
        """在Neo4j中创建约束和索引."""
        g = self.neo4j
        if g is None:
            raise RuntimeError("Neo4j not available")

        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Material) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Process) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Property) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Application) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Microstructure) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Composition) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Phase) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Paper) REQUIRE n.paper_id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Material) ON (n.material_class)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Property) ON (n.property_type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Process) ON (n.process_type)",
        ]
        for c in constraints:
            try:
                g.run(c)
            except Exception:
                pass
        for idx in indexes:
            try:
                g.run(idx)
            except Exception:
                pass
        print("  [Neo4j schema initialized]")

    # ================================================================
    # NetworkX 构建 (保持向后兼容)
    # ================================================================

    def build_from_extractions(self, parsed_papers: List[dict],
                                seed_kg: Optional[dict] = None) -> nx.MultiDiGraph:
        """从文献抽取结果构建NetworkX图."""
        if seed_kg:
            self._load_seed(seed_kg)
        for paper in parsed_papers:
            paper_id = paper.get("paper_id", "unknown")
            for entity in paper.get("entities", []):
                self._add_entity_nx(entity, paper_id)
            for relation in paper.get("relations", []):
                self._add_relation_nx(relation, paper_id)
        return self.graph

    def build_from_triplets(self, triplets: List[dict]) -> nx.MultiDiGraph:
        """从三元组列表构建NetworkX图."""
        for t in triplets:
            mat_name = t.get("material", "")
            prop_name = t.get("property", "")
            value = t.get("value", "")
            value_num = t.get("value_numeric")
            paper_id = t.get("paper_id", "")
            conf = t.get("confidence", 0.5)

            mat_id = self._get_or_create_node(mat_name, self._map_entity_type("material"), paper_id)
            prop_id = self._get_or_create_node(prop_name, self._map_entity_type("property"), paper_id)

            self.graph.add_edge(
                mat_id, prop_id,
                predicate="hasProperty",
                value=value,
                value_numeric=value_num,
                confidence=conf,
                paper_id=paper_id,
                evidence=t.get("evidence", ""),
            )

            proc_name = t.get("process")
            if proc_name:
                proc_id = self._get_or_create_node(proc_name, "Process", paper_id)
                self.graph.add_edge(mat_id, proc_id, predicate="usesProcess",
                                     confidence=conf, paper_id=paper_id)

            ms_name = t.get("microstructure")
            if ms_name:
                ms_id = self._get_or_create_node(ms_name, "Microstructure", paper_id)
                self.graph.add_edge(mat_id, ms_id, predicate="hasMicrostructure",
                                     confidence=conf, paper_id=paper_id)

        return self.graph

    def _map_entity_type(self, etype: str) -> str:
        """将旧版entity_type映射到Neo4j标签."""
        mapping = {
            "material": "Material",
            "property": "Property",
            "processing_method": "Process",
            "synthesis_method": "Process",
            "crystal_structure": "Microstructure",
            "microstructure": "Microstructure",
            "application": "Application",
            "property_value": "Property",
            "composition": "Composition",
            "phase": "Phase",
        }
        return mapping.get(etype, "Material")

    def _get_or_create_node(self, name: str, node_label: str,
                            paper_id: str = "") -> str:
        key = f"{name.lower()}|{node_label}"
        if key in self._entity_map:
            node_id = self._entity_map[key]
            if paper_id and paper_id not in self.graph.nodes[node_id].get("paper_ids", []):
                self.graph.nodes[node_id].setdefault("paper_ids", []).append(paper_id)
            return node_id
        node_id = self._make_id(f"{name}_{node_label}")
        self.graph.add_node(node_id,
            name=name, entity_type=node_label,
            source="extraction", paper_ids=[paper_id] if paper_id else [],
        )
        self._entity_map[key] = node_id
        return node_id

    def _load_seed(self, seed_kg: dict):
        for entity in seed_kg.get("entities", []):
            eid = entity.get("id", self._make_id(entity.get("name", "")))
            self.graph.add_node(eid,
                name=entity.get("name", eid),
                entity_type=entity.get("type", "Material"),
                source="seed", paper_ids=[],
            )
            self._entity_map[entity.get("name", "").lower()] = eid
        for rel in seed_kg.get("relations", []):
            subj_id = rel.get("subject")
            obj_id = rel.get("object")
            if subj_id and obj_id:
                self.graph.add_edge(subj_id, obj_id,
                    predicate=rel.get("predicate", "relatedTo"),
                    evidence=rel.get("evidence", ""), source="seed",
                )

    def _add_entity_nx(self, entity: dict, paper_id: str):
        name = entity.get("text", "")
        etype = entity.get("entity_type", "unknown")
        normalized = name.strip().lower()
        if normalized in self._entity_map:
            node_id = self._entity_map[normalized]
            pids = self.graph.nodes[node_id].get("paper_ids", [])
            if paper_id not in pids:
                pids.append(paper_id)
                self.graph.nodes[node_id]["paper_ids"] = pids
        else:
            node_id = self._make_id(f"{name}_{etype}")
            self.graph.add_node(node_id,
                name=name, entity_type=etype,
                source="extraction", paper_ids=[paper_id],
            )
            self._entity_map[normalized] = node_id

    def _add_relation_nx(self, relation: dict, paper_id: str):
        subj_name = relation.get("subject", "")
        obj_name = relation.get("object", "")
        predicate = relation.get("predicate", "relatedTo")
        subj_id = self._entity_map.get(subj_name.strip().lower())
        obj_id = self._entity_map.get(obj_name.strip().lower())
        if subj_id and obj_id and subj_id != obj_id:
            self.graph.add_edge(subj_id, obj_id,
                predicate=predicate,
                evidence=relation.get("evidence", ""),
                source="extraction", paper_id=paper_id,
            )

    # ================================================================
    # Neo4j 批量导入
    # ================================================================

    def import_to_neo4j(self, triplets: List[dict] = None,
                        parsed_papers: List[dict] = None) -> int:
        """将三元组/解析论文批量导入Neo4j.

        使用py2neo批量创建节点和关系.
        Returns: 创建的节点+关系总数
        """
        g = self.neo4j
        if g is None:
            raise RuntimeError("Neo4j not available. Check NEO4J_URI in config.")

        tx = g.begin()
        count = 0

        try:
            # 从三元组导入
            if triplets:
                for t in triplets:
                    mat_label = self._map_entity_type("material")
                    prop_label = self._map_entity_type("property")

                    mat_node = Node(mat_label, name=t.get("material", ""))
                    mat_node["source"] = "extraction"
                    mat_node["paper_ids"] = [t.get("paper_id", "")]

                    prop_node = Node(prop_label, name=t.get("property", ""))
                    prop_node["property_type"] = t.get("property", "")
                    prop_node["source"] = "extraction"

                    tx.merge(mat_node, mat_label, "name")
                    tx.merge(prop_node, prop_label, "name")

                    rel = Relationship(
                        mat_node, "hasProperty", prop_node,
                        value=t.get("value", ""),
                        value_numeric=t.get("value_numeric"),
                        confidence=t.get("confidence", 0.5),
                        paper_id=t.get("paper_id", ""),
                        evidence=t.get("evidence", "")[:500],
                    )
                    tx.create(rel)
                    count += 1

                    # 工艺节点
                    proc = t.get("process") or t.get("synthesis_method")
                    if proc:
                        proc_node = Node("Process", name=proc, source="extraction")
                        tx.merge(proc_node, "Process", "name")
                        tx.create(Relationship(mat_node, "usesProcess", proc_node,
                                               paper_id=t.get("paper_id", "")))
                        count += 1

                    # 微观结构节点
                    ms = t.get("microstructure")
                    if ms:
                        ms_node = Node("Microstructure", name=ms, source="extraction")
                        tx.merge(ms_node, "Microstructure", "name")
                        tx.create(Relationship(mat_node, "hasMicrostructure", ms_node,
                                               paper_id=t.get("paper_id", "")))
                        count += 1

            # 从解析论文导入
            if parsed_papers:
                for paper in parsed_papers:
                    pid = paper.get("paper_id", paper.get("filename", ""))
                    paper_node = Node("Paper",
                        paper_id=pid,
                        title=paper.get("title", ""),
                        year=paper.get("year", 0),
                        source="extraction",
                    )
                    tx.merge(paper_node, "Paper", "paper_id")
                    count += 1

                    for entity in paper.get("entities", []):
                        label = self._map_entity_type(entity.get("entity_type", ""))
                        e_node = Node(label, name=entity.get("text", ""),
                                       source="extraction")
                        tx.merge(e_node, label, "name")
                        tx.create(Relationship(paper_node, "reports", e_node,
                                               paper_id=pid))
                        count += 1

            g.commit(tx)
        except Exception as e:
            g.rollback(tx)
            raise RuntimeError(f"Neo4j import failed: {e}")

        return count

    def clear_neo4j(self):
        """清空Neo4j数据库 (谨慎使用)."""
        g = self.neo4j
        if g is None:
            raise RuntimeError("Neo4j not available")
        g.run("MATCH (n) DETACH DELETE n")

    # ================================================================
    # 通用
    # ================================================================

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

    @staticmethod
    def _make_id(seed: str) -> str:
        return f"node_{hashlib.md5(seed.encode()).hexdigest()[:10]}"
