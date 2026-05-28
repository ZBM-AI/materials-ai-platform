"""通用工具函数"""
import re
import json
import hashlib
from typing import List, Optional


def normalize_text(text: str) -> str:
    """规范化文本: 去多余空白、统一引号等"""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('“', '"').replace('”', '"')
    return text


def generate_id(prefix: str, seed: str) -> str:
    """基于内容的确定性ID生成"""
    digest = hashlib.md5(seed.encode('utf-8')).hexdigest()[:8]
    return f"{prefix}_{digest}"


def fuzzy_match(text: str, candidates: List[str], threshold: float = 0.8) -> Optional[str]:
    """简单模糊匹配: 返回最佳匹配候选词"""
    text_lower = text.lower().strip()
    best_match = None
    best_score = 0.0
    for cand in candidates:
        cand_lower = cand.lower().strip()
        if text_lower == cand_lower:
            return cand
        if cand_lower in text_lower or text_lower in cand_lower:
            score = min(len(text_lower), len(cand_lower)) / max(len(text_lower), len(cand_lower))
            if score > best_score:
                best_score = score
                best_match = cand
    if best_score >= threshold:
        return best_match
    return None


def load_json(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_divide(a, b):
    """安全除法，避免除零"""
    return a / b if b != 0 else 0.0
