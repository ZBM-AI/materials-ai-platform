"""材料科学实体识别 — 混合NER: spaCy + 正则 + 词典"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .patterns import (
    MATERIAL_NAMES, PROPERTY_NAMES, PROCESSING_KEYWORDS,
    CRYSTAL_STRUCTURES, APPLICATIONS, CHEMICAL_FORMULA_PATTERN,
    PROPERTY_VALUE_PATTERN, MATERIAL_KEYWORD_PATTERNS,
)
from utils.helpers import normalize_text, generate_id


@dataclass
class Entity:
    text: str
    entity_type: str
    start_char: int
    end_char: int
    sentence_idx: int = 0
    confidence: float = 1.0


class MaterialsNER:
    """混合实体识别: spaCy句子分割 + 正则(化学式) + 词典匹配"""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self._nlp = None
        self.spacy_model = spacy_model
        self._init_gazetteers()

    def _init_gazetteers(self):
        self.gazetteers = {
            "material": self._build_lookup(MATERIAL_NAMES),
            "property": self._build_lookup(PROPERTY_NAMES),
            "processing_method": self._build_lookup(PROCESSING_KEYWORDS),
            "crystal_structure": self._build_lookup(CRYSTAL_STRUCTURES),
            "application": self._build_lookup(APPLICATIONS),
        }
        self.material_keywords = re.compile(
            '|'.join(MATERIAL_KEYWORD_PATTERNS), re.IGNORECASE
        )

    def _build_lookup(self, term_list: list) -> dict:
        lookup = {}
        for term in term_list:
            key = term.lower().strip()
            lookup[key] = term
        return lookup

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self.spacy_model)
            except OSError:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", self.spacy_model], check=True)
                self._nlp = spacy.load(self.spacy_model)
        return self._nlp

    def extract_entities(self, text: str) -> List[Entity]:
        doc = self.nlp(text)
        entities: List[Entity] = []
        for sent_idx, sent in enumerate(doc.sents):
            sent_text = sent.text
            sent_start = sent.start_char
            entities.extend(self._match_chemical_formulas(sent_text, sent_start, sent_idx))
            entities.extend(self._match_gazetteer(sent_text, sent_start, sent_idx))
            entities.extend(self._match_property_values(sent_text, sent_start, sent_idx))
        return self._resolve_overlaps(entities)

    def _match_chemical_formulas(self, text: str, offset: int, sent_idx: int) -> List[Entity]:
        entities = []
        for m in re.finditer(CHEMICAL_FORMULA_PATTERN, text):
            formula = m.group()
            if not self._is_valid_formula(formula):
                continue
            if len(formula) < 2:
                continue
            common_words = {"The", "This", "We", "In", "On", "At", "By", "To", "Of", "Is",
                          "It", "Be", "No", "So", "As", "An", "Or", "He", "Do", "If",
                          "Fig", "Table", "Figure", "Eq", "Also", "Thus", "Such", "With",
                          "From", "For", "Has", "Are", "Was", "Can", "May", "Due", "Not"}
            if formula in common_words:
                continue
            entities.append(Entity(
                text=formula,
                entity_type="material",
                start_char=offset + m.start(),
                end_char=offset + m.end(),
                sentence_idx=sent_idx,
            ))
        return entities

    def _is_valid_formula(self, text: str) -> bool:
        if not re.match(r'^[A-Z][a-z]?\d*', text):
            return False
        has_uppercase = bool(re.search(r'[A-Z]', text))
        has_element_pattern = bool(re.search(r'[A-Z][a-z]?\d+', text))
        return has_uppercase and (has_element_pattern or len(text) <= 6)

    def _match_gazetteer(self, text: str, offset: int, sent_idx: int) -> List[Entity]:
        entities = []
        text_lower = text.lower()
        for entity_type, lookup in self.gazetteers.items():
            for term_key, term_display in lookup.items():
                pos = 0
                while True:
                    pos = text_lower.find(term_key, pos)
                    if pos == -1:
                        break
                    word_before_ok = pos == 0 or not text[pos - 1].isalnum()
                    word_after_ok = (pos + len(term_key) >= len(text) or
                                     not text[pos + len(term_key)].isalnum())
                    if word_before_ok and word_after_ok:
                        entities.append(Entity(
                            text=text[pos:pos + len(term_key)],
                            entity_type=entity_type,
                            start_char=offset + pos,
                            end_char=offset + pos + len(term_key),
                            sentence_idx=sent_idx,
                        ))
                    pos += max(len(term_key), 1)
        return entities

    def _match_property_values(self, text: str, offset: int, sent_idx: int) -> List[Entity]:
        entities = []
        for m in re.finditer(PROPERTY_VALUE_PATTERN, text):
            entities.append(Entity(
                text=m.group(),
                entity_type="property_value",
                start_char=offset + m.start(),
                end_char=offset + m.end(),
                sentence_idx=sent_idx,
            ))
        return entities

    def _resolve_overlaps(self, entities: List[Entity]) -> List[Entity]:
        if not entities:
            return entities
        # 优先: material > property > processing > structure > application > property_value
        priority = {
            "material": 6, "property": 5, "processing_method": 4,
            "crystal_structure": 3, "application": 2, "property_value": 1
        }
        sorted_entities = sorted(entities, key=lambda e: (e.start_char, -(e.end_char - e.start_char)))
        resolved = []
        for e in sorted_entities:
            conflict = False
            for r in resolved:
                if not (e.end_char <= r.start_char or e.start_char >= r.end_char):
                    if priority.get(e.entity_type, 0) > priority.get(r.entity_type, 0):
                        r.text = e.text
                        r.entity_type = e.entity_type
                        r.start_char = e.start_char
                        r.end_char = e.end_char
                        r.sentence_idx = e.sentence_idx
                        r.confidence = e.confidence
                    conflict = True
                    break
            if not conflict:
                resolved.append(e)
        return resolved
