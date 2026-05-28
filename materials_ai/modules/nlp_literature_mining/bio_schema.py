"""BIO标注格式定义 + token↔span转换 + WordPiece对齐工具

BIO标签体系 (6类实体):
  B/I-MAT: 材料 (TiO2, graphene oxide, FeCoNiCrMn)
  B/I-SYN: 合成方法 (sol-gel, CVD, ball milling)
  B/I-PRO: 性能参数 (band gap, tensile strength)
  B/I-VAL: 性能数值 (3.2 eV, 500 MPa)
  B/I-MIC: 微观结构 (perovskite, grain boundary)
  B/I-APP: 应用 (solar cell, Li-ion battery)

训练数据格式 (JSONL, 每行一个JSON):
  {"tokens": ["TiO2","has","a","band","gap","of","3.2","eV"],
   "tags":   ["B-MAT","O","O","B-PRO","I-PRO","O","B-VAL","I-VAL"]}
"""

from dataclasses import dataclass
from typing import List, Tuple
import config


@dataclass
class EntitySpan:
    text: str
    entity_type: str  # material, synthesis_method, property, ...
    start_token_idx: int
    end_token_idx: int  # inclusive


def tokens_to_spans(tokens: List[str], tags: List[str]) -> List[EntitySpan]:
    """BIO标签序列 → 实体span列表."""
    spans = []
    i = 0
    while i < len(tokens):
        tag = tags[i]
        if tag.startswith("B-"):
            prefix = tag[2:]
            etype = config.BIO_PREFIX_TO_TYPE.get(prefix, "material")
            start = i
            i += 1
            while i < len(tokens) and tags[i] == f"I-{prefix}":
                i += 1
            end = i - 1
            text = " ".join(tokens[start:end + 1])
            text = _reconstruct_text(tokens[start:end + 1])
            spans.append(EntitySpan(text=text, entity_type=etype,
                                    start_token_idx=start, end_token_idx=end))
        else:
            i += 1
    return spans


def spans_to_bio(tokens: List[str], spans: List[EntitySpan]) -> List[str]:
    """实体span列表 → BIO标签序列."""
    tags = ["O"] * len(tokens)
    for span in spans:
        prefix_map = {v: k for k, v in config.BIO_PREFIX_TO_TYPE.items()}
        prefix = prefix_map.get(span.entity_type, "MAT")
        tags[span.start_token_idx] = f"B-{prefix}"
        for i in range(span.start_token_idx + 1, span.end_token_idx + 1):
            tags[i] = f"I-{prefix}"
    return tags


def align_labels_with_wordpieces(
    tokens: List[str], tags: List[str], tokenizer
) -> Tuple[List[int], List[int]]:
    """将词级BIO标签对齐到WordPiece级.

    返回 (input_ids, label_ids). 特殊token ([CLS],[SEP]) 的标签 = -100 (忽略loss).
    子词 (##xx) 继承原词的B/I标签 (B改为I以保持一致性).
    """
    input_ids = []
    label_ids = []
    tag_to_id = config.TAG_TO_ID

    input_ids.append(tokenizer.cls_token_id)
    label_ids.append(-100)

    for token, tag in zip(tokens, tags):
        wp_ids = tokenizer.encode(token, add_special_tokens=False)
        if not wp_ids:
            continue
        input_ids.extend(wp_ids)
        tag_id = tag_to_id.get(tag, 0)
        label_ids.append(tag_id)
        for _ in range(len(wp_ids) - 1):
            if tag.startswith("B-"):
                i_tag = tag.replace("B-", "I-", 1)
                label_ids.append(tag_to_id.get(i_tag, tag_id))
            else:
                label_ids.append(tag_id)

    input_ids.append(tokenizer.sep_token_id)
    label_ids.append(-100)

    max_len = config.SCIBERT_MAX_LENGTH
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        label_ids = label_ids[:max_len]

    return input_ids, label_ids


def _reconstruct_text(tokens: List[str]) -> str:
    """从token列表重建原始文本 (处理##子词)."""
    result = []
    for tok in tokens:
        if tok.startswith("##"):
            result[-1] = result[-1] + tok[2:]
        else:
            result.append(tok)
    return " ".join(result)


def validate_bio_consistency(tokens: List[str], tags: List[str]) -> List[str]:
    """验证并修复BIO标签一致性. 返回修复后的标签列表."""
    if len(tokens) != len(tags):
        raise ValueError(f"Token/tag length mismatch: {len(tokens)} vs {len(tags)}")
    fixed = list(tags)
    i = 0
    while i < len(fixed):
        tag = fixed[i]
        if tag.startswith("I-") and (i == 0 or fixed[i - 1] == "O"):
            fixed[i] = tag.replace("I-", "B-", 1)
        if tag.startswith("I-") and i > 0 and fixed[i - 1] != "O":
            prev_prefix = fixed[i - 1][2:] if fixed[i - 1][0] in "BI" else ""
            curr_prefix = tag[2:]
            if prev_prefix and prev_prefix != curr_prefix:
                fixed[i] = f"B-{curr_prefix}"
        if tag.startswith("B-") and i > 0 and fixed[i - 1].startswith("B-"):
            pass  # valid: new entity starts
        i += 1
    return fixed
