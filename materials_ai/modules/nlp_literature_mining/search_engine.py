"""文献检索引擎 — 对提取的实体和关系进行搜索"""

import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    paper_id: str
    filename: str
    matched_entities: list
    matched_relations: list
    evidence_snippets: list


class LiteratureSearchEngine:
    def __init__(self, parsed_papers: List[dict]):
        self.papers = parsed_papers
        self._index = {}
        self._build_index()

    def _build_index(self):
        for paper in self.papers:
            pid = paper.get("paper_id", "")
            for entity in paper.get("entities", []):
                key = entity.get("text", "").lower()
                if key not in self._index:
                    self._index[key] = []
                self._index[key].append({
                    "paper_id": pid,
                    "entity": entity,
                })

    def search_entities(self, entity_type: str = None, keyword: str = None) -> List[dict]:
        results = []
        for paper in self.papers:
            for entity in paper.get("entities", []):
                type_match = entity_type is None or entity.get("entity_type") == entity_type
                kw_match = True
                if keyword:
                    kw_lower = keyword.lower()
                    kw_match = kw_lower in entity.get("text", "").lower()
                if type_match and kw_match:
                    results.append({
                        "paper_id": paper.get("paper_id"),
                        "filename": paper.get("filename"),
                        "entity": entity,
                    })
        return results

    def search_by_property_range(self, property_name: str,
                                  min_val: float, max_val: float) -> List[dict]:
        results = []
        value_pattern = re.compile(r'(\d+\.?\d*)\s*([eE][vV]|[mM][eE][vV]|[mM]?[pP]?[aA]|[gG][pP][aA])')
        for paper in self.papers:
            for relation in paper.get("relations", []):
                if (relation.get("predicate") == "hasValue" and
                        property_name.lower() in relation.get("subject", "").lower()):
                    obj_text = relation.get("object", "")
                    m = value_pattern.search(obj_text)
                    if m:
                        try:
                            val = float(m.group(1))
                            if min_val <= val <= max_val:
                                results.append({
                                    "paper_id": paper.get("paper_id"),
                                    "filename": paper.get("filename"),
                                    "relation": relation,
                                    "value": val,
                                })
                        except ValueError:
                            continue
        return results

    def find_related(self, entity_text: str) -> List[dict]:
        related = []
        target = entity_text.lower()
        for paper in self.papers:
            for relation in paper.get("relations", []):
                if (target in relation.get("subject", "").lower() or
                        target in relation.get("object", "").lower()):
                    related.append({
                        "paper_id": paper.get("paper_id"),
                        "filename": paper.get("filename"),
                        "relation": relation,
                    })
        return related

    def get_paper_summary(self, paper_id: str) -> Optional[dict]:
        for paper in self.papers:
            if paper.get("paper_id") == paper_id:
                return {
                    "paper_id": paper_id,
                    "filename": paper.get("filename"),
                    "num_entities": len(paper.get("entities", [])),
                    "num_relations": len(paper.get("relations", [])),
                    "entity_types": self._count_types(paper.get("entities", [])),
                    "abstract": paper.get("abstract", ""),
                }
        return None

    def get_statistics(self) -> dict:
        all_entities = []
        all_relations = []
        for paper in self.papers:
            all_entities.extend(paper.get("entities", []))
            all_relations.extend(paper.get("relations", []))
        entity_types = {}
        for e in all_entities:
            t = e.get("entity_type", "unknown")
            entity_types[t] = entity_types.get(t, 0) + 1
        relation_types = {}
        for r in all_relations:
            t = r.get("predicate", "unknown")
            relation_types[t] = relation_types.get(t, 0) + 1
        unique_materials = len(set(
            e.get("text") for e in all_entities if e.get("entity_type") == "material"
        ))
        return {
            "num_papers": len(self.papers),
            "num_entities": len(all_entities),
            "num_relations": len(all_relations),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "unique_materials": unique_materials,
        }

    def _count_types(self, entities: list) -> dict:
        counts = {}
        for e in entities:
            t = e.get("entity_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts
