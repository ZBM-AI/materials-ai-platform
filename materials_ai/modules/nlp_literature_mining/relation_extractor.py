"""关系抽取 — 基于spaCy依存分析的材料实体关系抽取 + 三元组提取"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .materials_ner import Entity


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    evidence: str = ""
    confidence: float = 1.0
    subject_type: str = ""
    object_type: str = ""


@dataclass
class Triplet:
    """材料-性能-数值 三元组."""
    material: str
    property: str
    value: str
    value_numeric: Optional[float] = None
    evidence: str = ""
    confidence: float = 0.5
    unit: str = ""


class RelationExtractor:
    """基于规则的依存分析关系抽取 + 三元组提取"""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self._nlp = None
        self.spacy_model = spacy_model
        self._value_pattern = re.compile(
            r'(\d+\.?\d*)\s*(eV|MPa|GPa|W/(?:m·K|mK)|W/mK|nm|μm|mm|%|K|°C|g/cm³|g/cm3|kg/m³|S/cm|S/m|W/cm·K)',
            re.IGNORECASE,
        )

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self.spacy_model)
            except OSError:
                import subprocess, sys
                subprocess.run([sys.executable, "-m", "spacy", "download", self.spacy_model], check=True)
                self._nlp = spacy.load(self.spacy_model)
        return self._nlp

    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        doc = self.nlp(text)
        relations = []
        for sent in doc.sents:
            sent_entities = [e for e in entities
                             if e.start_char >= sent.start_char and e.end_char <= sent.end_char]
            if len(sent_entities) < 2:
                continue
            relations.extend(self._extract_from_sentence(sent, sent_entities))
        return relations

    def extract_triplets(self, text: str, entities: List[Entity]) -> List[Triplet]:
        """从文本+实体中提取 (材料, 性能, 数值) 三元组."""
        doc = self.nlp(text)
        triplets = []
        for sent in doc.sents:
            sent_entities = [e for e in entities
                             if e.start_char >= sent.start_char and e.end_char <= sent.end_char]
            sent_triplets = self._extract_triplets_from_sentence(sent, sent_entities)
            triplets.extend(sent_triplets)
        return self._deduplicate_triplets(triplets)

    def _extract_from_sentence(self, sent, entities: List[Entity]) -> List[Relation]:
        relations = []
        material_entities = [e for e in entities if e.entity_type == "material"]
        property_entities = [e for e in entities if e.entity_type == "property"]
        processing_entities = [e for e in entities if e.entity_type in ("processing_method", "synthesis_method")]
        structure_entities = [e for e in entities if e.entity_type in ("crystal_structure", "microstructure")]
        application_entities = [e for e in entities if e.entity_type == "application"]
        property_values = [e for e in entities if e.entity_type == "property_value"]

        sent_text = sent.text

        for material in material_entities:
            for prop in property_entities:
                relations.append(Relation(
                    subject=material.text, predicate="hasProperty",
                    object=prop.text, evidence=sent_text,
                    subject_type="material", object_type="property",
                ))
            for proc in processing_entities:
                pred = "synthesizedBy" if proc.entity_type == "synthesis_method" else "processedBy"
                relations.append(Relation(
                    subject=material.text, predicate=pred,
                    object=proc.text, evidence=sent_text,
                    subject_type="material", object_type=proc.entity_type,
                ))
            for struct in structure_entities:
                pred = "hasMicrostructure" if struct.entity_type == "microstructure" else "hasStructure"
                relations.append(Relation(
                    subject=material.text, predicate=pred,
                    object=struct.text, evidence=sent_text,
                    subject_type="material", object_type=struct.entity_type,
                ))
            for app in application_entities:
                relations.append(Relation(
                    subject=material.text, predicate="usedIn",
                    object=app.text, evidence=sent_text,
                    subject_type="material", object_type="application",
                ))
            for val in property_values:
                for prop in property_entities:
                    relations.append(Relation(
                        subject=prop.text, predicate="hasValue",
                        object=val.text, evidence=sent_text,
                        subject_type="property", object_type="property_value",
                    ))

        for m1 in material_entities:
            for m2 in material_entities:
                if m1.text != m2.text:
                    relations.append(Relation(
                        subject=m1.text, predicate="relatedTo",
                        object=m2.text, evidence=sent_text,
                        subject_type="material", object_type="material",
                    ))

        return relations

    def _extract_triplets_from_sentence(self, sent, entities: List[Entity]) -> List[Triplet]:
        """从句中提取三元组: 在依存树中寻找 material←property→value 的路径."""
        triplets = []
        materials = [e for e in entities if e.entity_type == "material"]
        properties = [e for e in entities if e.entity_type == "property"]
        values = [e for e in entities if e.entity_type == "property_value"]

        sent_text = sent.text

        # 策略1: 距离优先 — 每个数值找最近的属性, 属性找最近的材料
        for val in values:
            best_prop = self._nearest_entity(val, properties)
            if best_prop is None:
                continue
            best_mat = self._nearest_entity(best_prop, materials)
            if best_mat is None:
                best_mat = self._nearest_entity(val, materials)

            value_str = val.text
            numeric, unit = self._parse_value_unit(value_str)

            confidence = 0.7
            if best_mat and self._distance(val, best_mat) < 100:
                confidence += 0.15
            if best_prop and self._distance(val, best_prop) < 50:
                confidence += 0.15

            triplets.append(Triplet(
                material=best_mat.text if best_mat else "unknown",
                property=best_prop.text,
                value=value_str,
                value_numeric=numeric,
                evidence=sent_text[:200],
                confidence=min(confidence, 1.0),
                unit=unit,
            ))

        # 策略2: 正则兜底 — value pattern + 最近的前序属性/材料
        for m in self._value_pattern.finditer(sent_text):
            val_text = m.group()
            val_start = sent.start_char + m.start()
            val_end = sent.start_char + m.end()

            already = any(
                t.evidence == sent_text[:200] and abs(
                    (sent_text.find(t.value) if t.value in sent_text else -1)
                ) == m.start()
                for t in triplets
            )
            if already:
                continue

            numeric = float(m.group(1))
            unit = m.group(2)
            val_virtual = Entity(text=val_text, entity_type="property_value",
                                  start_char=val_start, end_char=val_end)

            best_prop = self._nearest_entity(val_virtual, properties)
            best_mat = self._nearest_entity(val_virtual, materials)
            if best_mat is None and best_prop is not None:
                best_mat = self._nearest_entity(best_prop, materials)

            if best_prop:
                triplets.append(Triplet(
                    material=best_mat.text if best_mat else "unknown",
                    property=best_prop.text,
                    value=val_text,
                    value_numeric=numeric,
                    evidence=sent_text[:200],
                    confidence=0.5,
                    unit=unit,
                ))

        return triplets

    def _nearest_entity(self, target: Entity, candidates: List[Entity]) -> Optional[Entity]:
        """找到距离target最近的候选实体."""
        if not candidates:
            return None
        best = None
        best_dist = float('inf')
        for c in candidates:
            dist = min(abs(c.start_char - target.end_char), abs(c.end_char - target.start_char))
            if dist < best_dist:
                best_dist = dist
                best = c
        return best

    def _distance(self, a: Entity, b: Entity) -> int:
        return min(abs(a.start_char - b.end_char), abs(a.end_char - b.start_char))

    def _parse_value_unit(self, text: str) -> tuple:
        """从文本解析数值和单位."""
        m = re.search(r'([\d.]+)\s*([a-zA-Z/·℃²³]+)?', text)
        if m:
            try:
                numeric = float(m.group(1))
            except ValueError:
                numeric = None
            unit = m.group(2) or ""
            return numeric, unit
        return None, ""

    def _deduplicate_triplets(self, triplets: List[Triplet]) -> List[Triplet]:
        """去重: 相同(material, property, value)只保留置信度最高的."""
        seen = {}
        for t in triplets:
            key = (t.material.lower(), t.property.lower(), t.value.lower())
            if key not in seen or t.confidence > seen[key].confidence:
                seen[key] = t
        return sorted(seen.values(), key=lambda x: x.confidence, reverse=True)
