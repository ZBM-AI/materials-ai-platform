"""从已解析论文半自动生成BIO训练数据.

使用MaterialsNER提取实体 → 转为BIO标签 → 人工审核 → 保存为JSONL.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.nlp_literature_mining.materials_ner import MaterialsNER
from modules.nlp_literature_mining.bio_schema import validate_bio_consistency
from utils.data_loader import DataLoader
import config


def entities_to_bio(tokens: list, entities: list) -> list:
    """将MaterialsNER的Entity列表转为BIO标签序列.

    BIO标签映射:
      material → B/I-MAT
      property → B/I-PRO
      property_value → B/I-VAL
      synthesis_method → B/I-SYN
      processing_method → B/I-SYN
      microstructure → B/I-MIC
      crystal_structure → B/I-MIC
      application → B/I-APP
    """
    tags = ["O"] * len(tokens)

    # 构建字符位置→token索引映射
    char_to_token = []
    for i, tok in enumerate(tokens):
        for _ in range(len(tok)):
            char_to_token.append(i)

    type_to_prefix = {
        "material": "MAT",
        "property": "PRO",
        "property_value": "VAL",
        "synthesis_method": "SYN",
        "processing_method": "SYN",
        "microstructure": "MIC",
        "crystal_structure": "MIC",
        "application": "APP",
    }

    for ent in sorted(entities, key=lambda e: e.start_char):
        prefix = type_to_prefix.get(ent.entity_type)
        if prefix is None:
            continue

        start_char = ent.start_char
        end_char = ent.end_char

        if start_char >= len(char_to_token) or end_char > len(char_to_token):
            continue

        start_tok = char_to_token[start_char]
        end_tok = char_to_token[end_char - 1] if end_char > 0 else start_tok

        if start_tok >= len(tokens) or end_tok >= len(tokens):
            continue

        # 检查是否已有标签 (不覆盖)
        if tags[start_tok] != "O":
            continue

        tags[start_tok] = f"B-{prefix}"
        for i in range(start_tok + 1, end_tok + 1):
            if i < len(tokens) and tags[i] == "O":
                tags[i] = f"I-{prefix}"

    return tags


def simple_tokenize(text: str) -> list:
    """简单分词 (按空格/标点分割)."""
    import re
    tokens = []
    for m in re.finditer(r'[A-Za-z0-9]+(?:\.[0-9]+)?|[^\s\w]', text):
        tokens.append(m.group())
    return tokens


def main():
    parser = argparse.ArgumentParser(description="从论文生成BIO训练数据")
    parser.add_argument("--output", default=None,
                        help="输出JSONL路径 (默认: data/ner_training/seed_bio_data.jsonl)")
    parser.add_argument("--max-papers", type=int, default=20,
                        help="处理的最大论文数")
    args = parser.parse_args()

    output_path = args.output or os.path.join(config.NER_TRAINING_DIR, "seed_bio_data.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ner = MaterialsNER()
    papers = DataLoader.load_parsed_papers()

    if not papers:
        print("未找到已解析论文。请先解析一些论文。")
        return

    print(f"处理 {min(len(papers), args.max_papers)} 篇论文...")

    records = []
    for i, paper in enumerate(papers[:args.max_papers]):
        text = paper.get("raw_text", "")
        if not text:
            continue

        # 按句子分割
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if len(s.strip()) > 20]

        for sent_idx, sent in enumerate(sentences[:30]):  # 每篇最多30句
            tokens = simple_tokenize(sent)
            if len(tokens) < 5:
                continue
            entities = ner.extract_entities(sent)
            if not entities:
                continue

            tags = entities_to_bio(tokens, entities)
            tags = validate_bio_consistency(tokens, tags)

            # 只保留有至少一个实体的句子
            if all(t == "O" for t in tags):
                continue

            records.append({"tokens": tokens, "tags": tags})

        if (i + 1) % 5 == 0:
            print(f"  已处理 {i + 1} 篇论文, 累计 {len(records)} 条标注句子")

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"保存 {len(records)} 条BIO标注数据到 {output_path}")

    # 输出标签分布
    tag_counts = {}
    for rec in records:
        for tag in rec["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print("\n标签分布:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
