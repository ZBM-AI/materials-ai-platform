"""SciBERT NER微调训练pipeline — Dataset/Dataloader/Trainer"""

import os
import json
import numpy as np
from typing import List, Dict, Optional

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        TrainingArguments, Trainer,
        DataCollatorForTokenClassification,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from .bio_schema import align_labels_with_wordpieces


class MaterialsNERDataset(Dataset):
    """BIO标注材料科学NER数据集."""

    def __init__(self, data: List[dict], tokenizer, max_length: int = 512,
                 tag_to_id: Dict[str, int] = None):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers/torch not installed")
        if tag_to_id is None:
            from config import TAG_TO_ID
            tag_to_id = TAG_TO_ID
        self.encodings = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in data:
            tokens = item.get("tokens", [])
            tags = item.get("tags", [])
            input_ids, label_ids = align_labels_with_wordpieces(
                tokens, tags, tokenizer
            )
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                label_ids = label_ids[:max_length]
            attn_mask = [1] * len(input_ids)
            while len(input_ids) < max_length:
                input_ids.append(tokenizer.pad_token_id or 0)
                attn_mask.append(0)
                label_ids.append(-100)
            self.encodings["input_ids"].append(input_ids)
            self.encodings["attention_mask"].append(attn_mask)
            self.encodings["labels"].append(label_ids)

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        return {key: torch.tensor(val[idx], dtype=torch.long)
                for key, val in self.encodings.items()}


class SciBERTNERTrainer:
    """SciBERT微调pipeline: 加载数据 → 训练 → 评估 → 保存."""

    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased",
                 output_dir: str = None, max_length: int = 512,
                 batch_size: int = 8, learning_rate: float = 2e-5,
                 num_epochs: int = 3, tag_to_id: Dict[str, int] = None):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers/torch not installed. "
                "Run: pip install torch transformers datasets seqeval accelerate"
            )
        self.model_name = model_name
        self.output_dir = output_dir or "saved_models/scibert_ner"
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.tag_to_id = tag_to_id
        if tag_to_id is None:
            from config import TAG_TO_ID
            self.tag_to_id = TAG_TO_ID

        self.tokenizer = None
        self.model = None
        self.id_to_tag = {v: k for k, v in self.tag_to_id.items()}

    def load_data(self, jsonl_path: str) -> List[dict]:
        """从JSONL文件加载BIO标注数据."""
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def prepare_dataset(self, data: List[dict]) -> MaterialsNERDataset:
        """将原始数据转为PyTorch Dataset."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return MaterialsNERDataset(data, self.tokenizer, self.max_length, self.tag_to_id)

    def train(self, train_data: List[dict], val_split: float = 0.1,
              save_best: bool = True):
        """完整训练流程."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # 分割
        split = max(1, int(len(train_data) * val_split))
        val_data = train_data[-split:] if split > 0 else []
        train_sub = train_data[:-split] if split > 0 else train_data

        train_dataset = self.prepare_dataset(train_sub)
        val_dataset = self.prepare_dataset(val_data) if val_data else None

        # 模型
        id_to_tag = self.id_to_tag
        num_labels = len(id_to_tag)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, num_labels=num_labels,
            id2label=id_to_tag, label2id=self.tag_to_id,
        )

        # 训练配置
        args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch" if save_best else "no",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            logging_steps=10,
            load_best_model_at_end=save_best and val_dataset is not None,
            metric_for_best_model="eval_f1" if val_dataset else None,
            report_to="none",
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics if val_dataset else None,
        )

        trainer.train()

        if save_best:
            self.save_model(self.output_dir)

        return trainer

    def evaluate(self, test_data: List[dict]) -> dict:
        """在测试集上评估, 返回seqeval分类报告."""
        if self.model is None:
            try:
                self.model = AutoModelForTokenClassification.from_pretrained(self.output_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
            except Exception:
                raise RuntimeError("No model loaded. Train or load a model first.")

        from seqeval.metrics import classification_report
        test_dataset = self.prepare_dataset(test_data)
        y_true, y_pred = [], []

        self.model.eval()
        with torch.no_grad():
            for i in range(len(test_dataset)):
                item = test_dataset[i]
                input_ids = item["input_ids"].unsqueeze(0)
                attention_mask = item["attention_mask"].unsqueeze(0)
                labels = item["labels"]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1).squeeze(0)

                true_tags = []
                pred_tags = []
                for j, (lbl, pred) in enumerate(zip(labels, preds)):
                    if lbl.item() != -100:
                        true_tags.append(self.id_to_tag.get(lbl.item(), "O"))
                        pred_tags.append(self.id_to_tag.get(pred.item(), "O"))
                y_true.append(true_tags)
                y_pred.append(pred_tags)

        return classification_report(y_true, y_pred, output_dict=True)

    def _compute_metrics(self, eval_pred):
        """seqeval指标计算 (供Trainer使用)."""
        from seqeval.metrics import classification_report, f1_score
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        y_true, y_pred = [], []
        for pred_seq, label_seq in zip(predictions, labels):
            true_tags, pred_tags = [], []
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    true_tags.append(self.id_to_tag.get(l, "O"))
                    pred_tags.append(self.id_to_tag.get(p, "O"))
            y_true.append(true_tags)
            y_pred.append(pred_tags)

        return {
            "f1": f1_score(y_true, y_pred),
            "precision": float(classification_report(y_true, y_pred, output_dict=True)
                              .get("weighted avg", {}).get("precision", 0)),
            "recall": float(classification_report(y_true, y_pred, output_dict=True)
                           .get("weighted avg", {}).get("recall", 0)),
        }

    def save_model(self, output_dir: str):
        """保存模型和tokenizer."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model to save.")
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @staticmethod
    def load_model(model_dir: str):
        """加载已保存模型."""
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        return model, tokenizer


# ================================================================
# Standalone utilities: BIO data generation (used by Streamlit UI)
# ================================================================

def generate_seed_bio_data(ner, papers: list, max_papers: int = 10, output_path: str = None) -> tuple:
    """使用规则引擎自动标注论文，生成BIO种子训练数据.

    Args:
        ner: MaterialsNER 实例
        papers: 已解析论文列表
        max_papers: 最大处理论文数
        output_path: 输出JSONL路径

    Returns:
        (records_count, tag_counts_dict, output_path)
    """
    import re as _re
    from .bio_schema import validate_bio_consistency
    import config as _config

    records = []
    type_map = {
        "material": "MAT", "property": "PRO", "property_value": "VAL",
        "synthesis_method": "SYN", "processing_method": "SYN",
        "microstructure": "MIC", "crystal_structure": "MIC", "application": "APP",
    }

    for paper in papers[:max_papers]:
        text = paper.get("raw_text", "")
        if not text:
            continue
        for sent in [s.strip() for s in text.replace("\n", " ").split(". ") if len(s.strip()) > 20][:30]:
            tokens = [_m.group() for _m in _re.finditer(r'[A-Za-z0-9]+(?:\.[0-9]+)?|[^\s\w]', sent)]
            if len(tokens) < 5:
                continue
            entities = ner.extract_entities(sent)
            if not entities:
                continue
            tags = ["O"] * len(tokens)
            char_to_token = []
            for ti, tok in enumerate(tokens):
                for _ in range(len(tok)):
                    char_to_token.append(ti)
            for ent in sorted(entities, key=lambda e: e.start_char):
                prefix = type_map.get(ent.entity_type)
                if not prefix:
                    continue
                sc, ec = ent.start_char, ent.end_char
                if sc >= len(char_to_token) or ec > len(char_to_token):
                    continue
                st_tok, et_tok = char_to_token[sc], char_to_token[ec - 1] if ec > 0 else sc
                if st_tok >= len(tokens) or et_tok >= len(tokens):
                    continue
                if tags[st_tok] != "O":
                    continue
                tags[st_tok] = f"B-{prefix}"
                for i in range(st_tok + 1, et_tok + 1):
                    if i < len(tokens) and tags[i] == "O":
                        tags[i] = f"I-{prefix}"
            tags = validate_bio_consistency(tokens, tags)
            if any(t != "O" for t in tags):
                records.append({"tokens": tokens, "tags": tags})

    if output_path is None:
        import config as _c
        output_path = os.path.join(_c.NER_TRAINING_DIR, "seed_bio_data.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tag_counts = {}
    for rec in records:
        for tag in rec["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return len(records), tag_counts, output_path


# Synthetic data: templates and entity pools
_SYNTH_TEMPLATES = [
    "{MAT} was synthesized by {SYN} method.",
    "{MAT} nanoparticles were prepared via {SYN}.",
    "{MAT} exhibited a {PRO} of {VAL}.",
    "The {PRO} of {MAT} was measured to be {VAL}.",
    "{MAT} showed {MIC} structure.",
    "{MAT} is promising for {APP}.",
    "{MAT} was synthesized by {SYN} and showed {MIC} structure.",
    "{MAT} prepared by {SYN} achieved a {PRO} of {VAL}.",
    "For {APP}, {MAT} synthesized via {SYN} demonstrated {PRO} of {VAL}.",
    "We report {MAT} with {MIC} structure for {APP}, achieving {PRO} of {VAL}.",
    "A novel {SYN} route yielded {MAT} with superior {PRO} ({VAL}).",
    "The {SYN} process was used to fabricate {MAT} thin films.",
    "{MAT} with {MIC} morphology was obtained.",
    "{MAT} was applied in {APP} devices.",
    "{MAT} doped with {MAT} showed improved {PRO}.",
    "The composite of {MAT} and {MAT} exhibited enhanced {PRO}.",
    "Structural analysis revealed {MIC} phase in {MAT} after {SYN}.",
    "Optimizing {SYN} parameters improved {PRO} of {MAT} to {VAL}.",
    "The {PRO} was determined to be {VAL} for {MAT} prepared by {SYN}.",
    "Electrochemical tests showed {PRO} values of {VAL} for {MAT}.",
]

_SYNTH_MATERIALS = [
    "TiO2", "ZnO", "Fe2O3", "Al2O3", "SiO2", "ZrO2", "CeO2", "MgO",
    "CuO", "NiO", "Co3O4", "MnO2", "SnO2", "ITO", "BaTiO3", "SrTiO3",
    "graphene", "graphene oxide", "rGO", "carbon nanotube", "CNT",
    "MoS2", "WS2", "h-BN", "black phosphorus", "C60",
    "perovskite", "MAPbI3", "CsPbI3", "FAPbI3",
    "LCO", "LFP", "NMC", "NCA", "LMO", "graphite", "LLZO",
    "bismuth telluride", "lead telluride", "skutterudite",
    "g-C3N4", "MOF", "ZSM-5", "zeolite",
    "steel", "aluminum alloy", "titanium alloy", "HEA", "FeCoNiCrMn",
    "silicon", "GaAs", "GaN", "SiC", "CdTe",
]

_SYNTH_METHODS = [
    "sol-gel", "hydrothermal", "CVD", "PVD", "ball milling",
    "electrodeposition", "sputtering", "spin coating", "dip coating",
    "spray pyrolysis", "co-precipitation", "solid-state reaction",
    "mechanochemical", "microwave-assisted", "ultrasonic",
    "atomic layer deposition", "molecular beam epitaxy", "PLD",
    "electrospinning", "template-assisted", "self-assembly",
    "calcination", "annealing", "quenching", "hot pressing",
    "spark plasma sintering", "3D printing", "laser ablation",
]

_SYNTH_PROPERTIES = [
    "band gap", "electrical conductivity", "thermal conductivity",
    "tensile strength", "yield strength", "hardness",
    "dielectric constant", "refractive index", "carrier mobility",
    "specific capacity", "energy density", "power density",
    "Seebeck coefficient", "figure of merit", "ZT",
    "elastic modulus", "bulk modulus", "shear modulus",
    "fracture toughness", "wear resistance", "corrosion resistance",
    "surface area", "porosity", "catalytic activity",
]

_SYNTH_VALUES = [
    "3.2 eV", "1.8 eV", "2.5 eV", "500 MPa", "1.2 GPa", "350 MPa",
    "1200 MPa", "250 GPa", "80 GPa", "15 GPa", "450 HV", "650 HV",
    "150 W/mK", "0.5 W/mK", "300 W/mK", "200 mAh/g", "150 mAh/g",
    "300 F/g", "0.8", "1.5", "99.9%", "2000 m2/g", "50 nm", "100 nm",
]

_SYNTH_MICROSTRUCTURES = [
    "perovskite", "spinel", "amorphous", "crystalline",
    "nanocrystalline", "polycrystalline", "grain boundary",
    "nanoparticle", "nanorod", "nanowire", "nanosheet",
    "core-shell", "hollow sphere", "mesoporous", "hierarchical",
    "dendrite", "lamellar", "columnar", "equiaxed",
    "precipitate", "dislocation", "twin boundary",
]

_SYNTH_APPLICATIONS = [
    "solar cell", "Li-ion battery", "supercapacitor", "fuel cell",
    "thermoelectric generator", "photodetector", "LED",
    "gas sensor", "biosensor", "catalyst", "electrocatalyst",
    "photocatalyst", "water splitting", "CO2 reduction",
    "superconductor", "magnetic storage", "piezoelectric device",
    "drug delivery", "tissue engineering", "corrosion protection",
    "thermal barrier coating", "wear-resistant coating",
]

_SYNTH_POOLS = {
    "{MAT}": _SYNTH_MATERIALS, "{SYN}": _SYNTH_METHODS,
    "{PRO}": _SYNTH_PROPERTIES, "{VAL}": _SYNTH_VALUES,
    "{MIC}": _SYNTH_MICROSTRUCTURES, "{APP}": _SYNTH_APPLICATIONS,
}

_BIO_PREFIX_MAP = {"{MAT}": "MAT", "{SYN}": "SYN", "{PRO}": "PRO",
                   "{VAL}": "VAL", "{MIC}": "MIC", "{APP}": "APP"}


def generate_synthetic_bio_data(n_samples: int = 1000, output_path: str = None,
                                random_seed: int = 42) -> tuple:
    """使用词典模板生成合成BIO训练数据.

    Args:
        n_samples: 生成样本数
        output_path: 输出JSONL路径
        random_seed: 随机种子

    Returns:
        (records_count, tag_counts_dict, output_path)
    """
    import random as _random
    from .bio_schema import validate_bio_consistency

    _random.seed(random_seed)
    records = []

    for _ in range(n_samples):
        template = _random.choice(_SYNTH_TEMPLATES)
        placeholders = {}
        for ph, pool in _SYNTH_POOLS.items():
            if ph in template:
                placeholders[ph] = [_random.choice(pool) for _ in range(template.count(ph))]
        sentence = template
        for ph, vals in placeholders.items():
            for v in vals:
                sentence = sentence.replace(ph, v, 1)
        tokens = sentence.split()
        tags = ["O"] * len(tokens)
        for ph, vals in placeholders.items():
            pref = _BIO_PREFIX_MAP[ph]
            for v in vals:
                v_toks = v.split()
                for ti in range(len(tokens) - len(v_toks) + 1):
                    if tokens[ti:ti + len(v_toks)] == v_toks:
                        tags[ti] = f"B-{pref}"
                        for j in range(1, len(v_toks)):
                            tags[ti + j] = f"I-{pref}"
                        break
        tags = validate_bio_consistency(tokens, tags)
        if any(t != "O" for t in tags):
            records.append({"tokens": tokens, "tags": tags})

    if output_path is None:
        import config as _c
        output_path = os.path.join(_c.NER_TRAINING_DIR, "augmented_bio_data.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tag_counts = {}
    for rec in records:
        for tag in rec["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return len(records), tag_counts, output_path


# BIO visualization color map
BIO_COLOR_MAP = {
    "B-MAT": "#FF6B6B", "I-MAT": "#FF8E8E",
    "B-SYN": "#4ECDC4", "I-SYN": "#7EDDD6",
    "B-PRO": "#45B7D1", "I-PRO": "#73CCE0",
    "B-VAL": "#96CEB4", "I-VAL": "#B5DCC8",
    "B-MIC": "#FFEAA7", "I-MIC": "#FFF0C4",
    "B-APP": "#DDA0DD", "I-APP": "#E8C0E8",
}


def render_bio_preview_html(tokens: list, tags: list) -> str:
    """将BIO标注的token列表渲染为带颜色的HTML."""
    parts = []
    for tok, tag in zip(tokens, tags):
        if tag == "O":
            parts.append(f'<span style="color:#888">{tok}</span>')
        else:
            bg = BIO_COLOR_MAP.get(tag, "#CCC")
            parts.append(
                f'<span style="background:{bg};color:#000;padding:1px 3px;'
                f'border-radius:3px;margin:0 1px;font-weight:bold" '
                f'title="{tag}">{tok}</span>'
            )
    return (
        '<div style="font-family:monospace;font-size:13px;line-height:2;word-break:break-all">'
        + " ".join(parts) + "</div>"
    )
