"""SciBERT NER推理引擎 — 加载微调模型 + 自动回退到规则MaterialsNER"""

import os
import re
from typing import List, Optional

import config
from .materials_ner import Entity, MaterialsNER
from .bio_schema import tokens_to_spans


class SciBERTNER:
    """SciBERT微调模型推理器.

    加载已训练的SciBERT token classification模型进行NER.
    如果模型不可用, 自动回退到基于规则的MaterialsNER.
    """

    def __init__(self, model_dir: str = None, spacy_model: str = "en_core_web_sm"):
        self.model_dir = model_dir or config.NER_MODEL_DIR
        self.spacy_model = spacy_model
        self._model = None
        self._tokenizer = None
        self._fallback = None
        self._id_to_tag = config.ID_TO_TAG
        self._tag_to_id = config.TAG_TO_ID
        self._loaded = False
        self._load_attempted = False

    @property
    def use_fallback(self) -> bool:
        self._ensure_loaded()
        return self._model is None

    def _ensure_loaded(self):
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            if os.path.exists(self.model_dir):
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                self._model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
                self._loaded = True
        except Exception:
            self._model = None
            self._tokenizer = None

    def _get_fallback(self) -> MaterialsNER:
        if self._fallback is None:
            self._fallback = MaterialsNER(spacy_model=self.spacy_model)
        return self._fallback

    def extract_entities(self, text: str) -> List[Entity]:
        """从文本中提取材料科学实体. 自动选择SciBERT或规则引擎."""
        self._ensure_loaded()
        if self._model is not None and self._tokenizer is not None:
            return self._extract_with_scibert(text)
        return self._get_fallback().extract_entities(text)

    def _extract_with_scibert(self, text: str) -> List[Entity]:
        """使用SciBERT模型进行NER推理."""
        import torch

        sentences = self._split_sentences(text)
        all_entities = []

        for sent_text, sent_start in sentences:
            tokens, char_offsets = self._tokenize_with_offsets(sent_text)
            if not tokens:
                continue

            input_ids = [self._tokenizer.cls_token_id] + self._tokenizer.convert_tokens_to_ids(tokens) + [self._tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            max_len = config.SCIBERT_MAX_LENGTH

            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]

            with torch.no_grad():
                outputs = self._model(
                    input_ids=torch.tensor([input_ids], dtype=torch.long),
                    attention_mask=torch.tensor([attention_mask], dtype=torch.long),
                )
                predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0)

            # 去掉[CLS]和[SEP], 只保留实际token的预测
            pred_ids = predictions[1:len(tokens) + 1].tolist()
            tags = [self._id_to_tag.get(pid, "O") for pid in pred_ids]

            # BIO tag序列 → EntitySpan列表, 再转成Entity
            spans = tokens_to_spans(tokens, tags)
            for span in spans:
                start_char = char_offsets[span.start_token_idx][0]
                end_char = char_offsets[span.end_token_idx][1]
                all_entities.append(Entity(
                    text=span.text,
                    entity_type=span.entity_type,
                    start_char=sent_start + start_char,
                    end_char=sent_start + end_char,
                    confidence=0.85,
                ))

        return self._resolve_overlaps(all_entities)

    def _tokenize_with_offsets(self, text: str):
        """Tokenize并记录每个WordPiece token的字符偏移."""
        encoded = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        wp_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        tokens = []
        char_offsets = []
        for wp_id, (start, end) in zip(wp_ids, offsets):
            wp_token = self._tokenizer.convert_ids_to_tokens([wp_id])[0]
            tokens.append(wp_token)
            char_offsets.append((start, end))
        return tokens, char_offsets

    def _split_sentences(self, text: str) -> List[tuple]:
        """简单分句, 返回[(sentence_text, start_char), ...]."""
        sentences = []
        for m in re.finditer(r'[^.!?\n]+[.!?\n]?', text):
            sent = m.group().strip()
            if sent and len(sent) > 2:
                sentences.append((sent, m.start()))
        if not sentences:
            text_stripped = text.strip()
            if text_stripped:
                sentences.append((text_stripped, 0))
        return sentences

    def _resolve_overlaps(self, entities: List[Entity]) -> List[Entity]:
        """解决重叠实体 (与MaterialsNER逻辑一致的优先级)."""
        if not entities:
            return entities
        priority = {
            "material": 6, "property": 5, "synthesis_method": 4,
            "microstructure": 3, "application": 2, "property_value": 1,
            "processing_method": 4, "crystal_structure": 3,
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
                        r.confidence = e.confidence
                    conflict = True
                    break
            if not conflict:
                resolved.append(e)
        return resolved
