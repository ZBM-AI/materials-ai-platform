"""从词典模板生成合成BIO训练数据 — 增加训练数据量和多样性."""

import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.nlp_literature_mining.bio_schema import validate_bio_consistency
import config

# 模板: {MAT}, {SYN}, {PRO}, {VAL}, {MIC}, {APP} 占位符
TEMPLATES = [
    "{MAT} was synthesized by {SYN} method.",
    "{MAT} nanoparticles were prepared via {SYN}.",
    "The {SYN} process was used to fabricate {MAT} thin films.",
    "{MAT} exhibited a {PRO} of {VAL}.",
    "The {PRO} of {MAT} was measured to be {VAL}.",
    "A {PRO} was observed for {MAT}, reaching {VAL}.",
    "{MAT} showed {MIC} structure.",
    "The {MIC} phase was identified in {MAT}.",
    "{MAT} with {MIC} morphology was obtained.",
    "{MAT} is promising for {APP}.",
    "{MAT} was applied in {APP} devices.",
    "The {APP} performance was enhanced by {MAT}.",
    "{MAT} was synthesized by {SYN} and showed {MIC} structure.",
    "{MAT} prepared by {SYN} achieved a {PRO} of {VAL}.",
    "For {APP}, {MAT} synthesized via {SYN} demonstrated {PRO} of {VAL}.",
    "The {MIC} {MAT} prepared through {SYN} was evaluated for {APP}.",
    "{MAT} fabricated by {SYN} exhibited {PRO} of {VAL} with {MIC} microstructure.",
    "We report {MAT} with {MIC} structure for {APP}, achieving {PRO} of {VAL}.",
    "Using {SYN}, {MAT} was produced with {MIC} characteristics suitable for {APP}.",
    "The {PRO} of {MAT} ({VAL}) makes it a candidate for {APP}.",
    "A novel {SYN} route yielded {MAT} with superior {PRO} ({VAL}).",
    "{MAT} doped with {MAT} showed improved {PRO}.",
    "The composite of {MAT} and {MAT} exhibited enhanced {PRO}.",
    "Compared to pure {MAT}, the doped sample showed {MIC} features.",
    "{MAT} was deposited on substrate via {SYN}, then characterized by XRD and SEM.",
    "Structural analysis revealed {MIC} phase in {MAT} after {SYN}.",
    "Optimizing {SYN} parameters improved {PRO} of {MAT} to {VAL}.",
    "The combination of {MAT} and {SYN} resulted in {MIC} morphology.",
    "The {PRO} was determined to be {VAL} for {MAT} prepared by {SYN}.",
    "Electrochemical tests showed {PRO} values of {VAL} for {MAT}.",
]

# 实体填充值 (从patterns.py提取常用实体)
MATERIALS = [
    "TiO2", "ZnO", "Fe2O3", "Al2O3", "SiO2", "ZrO2", "CeO2", "MgO",
    "CuO", "NiO", "Co3O4", "MnO2", "SnO2", "ITO", "BaTiO3", "SrTiO3",
    "graphene", "graphene oxide", "rGO", "carbon nanotube", "CNT",
    "MoS2", "WS2", "h-BN", "black phosphorus", "C60",
    "perovskite", "MAPbI3", "CsPbI3", "FAPbI3", "double perovskite",
    "LCO", "LFP", "NMC", "NCA", "LMO", "graphite", "LLZO",
    "bismuth telluride", "lead telluride", "skutterudite", "half-Heusler",
    "g-C3N4", "MOF", "ZSM-5", "zeolite",
    "steel", "stainless steel", "aluminum alloy", "titanium alloy",
    "magnesium alloy", "nickel superalloy", "HEA", "FeCoNiCrMn",
    "silicon", "GaAs", "GaN", "SiC", "InP", "CdTe",
    "Si3N4", "boron carbide", "tungsten carbide", "hydroxyapatite",
]

SYNTHESIS_METHODS = [
    "sol-gel", "hydrothermal", "CVD", "PVD", "ball milling",
    "electrodeposition", "sputtering", "spin coating", "dip coating",
    "spray pyrolysis", "co-precipitation", "solid-state reaction",
    "mechanochemical", "microwave-assisted", "ultrasonic",
    "atomic layer deposition", "molecular beam epitaxy", "PLD",
    "electrospinning", "template-assisted", "self-assembly",
    "calcination", "annealing", "quenching", "hot pressing",
    "spark plasma sintering", "3D printing", "laser ablation",
]

PROPERTIES = [
    "band gap", "electrical conductivity", "thermal conductivity",
    "tensile strength", "yield strength", "hardness",
    "dielectric constant", "refractive index", "carrier mobility",
    "specific capacity", "energy density", "power density",
    "Seebeck coefficient", "figure of merit", "ZT",
    "elastic modulus", "bulk modulus", "shear modulus",
    "fracture toughness", "wear resistance", "corrosion resistance",
    "surface area", "porosity", "catalytic activity",
    "photoluminescence", "quantum yield", "transmittance",
]

VALUES = [
    "3.2 eV", "1.8 eV", "2.5 eV", "500 MPa", "1.2 GPa", "350 MPa",
    "1200 MPa", "250 GPa", "80 GPa", "15 GPa", "450 HV", "650 HV",
    "150 W/mK", "0.5 W/mK", "300 W/mK", "10 S/cm", "0.01 S/cm",
    "200 mAh/g", "150 mAh/g", "300 F/g", "5.8 W/(m·K)", "0.3 eV",
    "200 μV/K", "0.8", "1.5", "99.9%", "50%", "2000 m²/g",
    "50 nm", "100 nm", "10 μm", "5 μm",
]

MICROSTRUCTURES = [
    "perovskite", "spinel", "amorphous", "crystalline",
    "nanocrystalline", "polycrystalline", "grain boundary",
    "nanoparticle", "nanorod", "nanowire", "nanosheet",
    "core-shell", "hollow sphere", "mesoporous", "hierarchical",
    "dendrite", "lamellar", "columnar", "equiaxed",
    "precipitate", "dislocation", "twin boundary",
]

APPLICATIONS = [
    "solar cell", "Li-ion battery", "supercapacitor", "fuel cell",
    "thermoelectric generator", "photodetector", "LED",
    "gas sensor", "biosensor", "catalyst", "electrocatalyst",
    "photocatalyst", "water splitting", "CO2 reduction",
    "superconductor", "magnetic storage", "piezoelectric device",
    "drug delivery", "tissue engineering", "corrosion protection",
    "thermal barrier coating", "wear-resistant coating",
]


def generate_synthetic_data(n_samples: int = 2000) -> list:
    """生成合成BIO标注数据."""
    records = []
    for _ in range(n_samples):
        template = random.choice(TEMPLATES)

        # 收集占位符及其实例值
        placeholders = {}
        if "{MAT}" in template:
            count = template.count("{MAT}")
            placeholders["MAT"] = random.sample(MATERIALS, min(count, len(MATERIALS)))
        if "{SYN}" in template:
            placeholders["SYN"] = [random.choice(SYNTHESIS_METHODS)]
        if "{PRO}" in template:
            placeholders["PRO"] = [random.choice(PROPERTIES)]
        if "{VAL}" in template:
            placeholders["VAL"] = [random.choice(VALUES)]
        if "{MIC}" in template:
            placeholders["MIC"] = [random.choice(MICROSTRUCTURES)]
        if "{APP}" in template:
            placeholders["APP"] = [random.choice(APPLICATIONS)]

        # 填充模板
        sentence = template
        for ptype, vals in placeholders.items():
            for val in vals:
                sentence = sentence.replace("{" + ptype + "}", val, 1)

        # 分词
        tokens = sentence.split()
        tags = ["O"] * len(tokens)

        # 为每个填充值标注BIO
        consumed = {k: 0 for k in placeholders}
        for ptype, vals in placeholders.items():
            prefix_map = {
                "MAT": "MAT", "SYN": "SYN", "PRO": "PRO",
                "VAL": "VAL", "MIC": "MIC", "APP": "APP",
            }
            prefix = prefix_map[ptype]
            for val in vals:
                val_tokens = val.split()
                # 在tokens中找到val的位置
                for i in range(len(tokens) - len(val_tokens) + 1):
                    if tokens[i:i + len(val_tokens)] == val_tokens:
                        tags[i] = f"B-{prefix}"
                        for j in range(1, len(val_tokens)):
                            tags[i + j] = f"I-{prefix}"
                        break

        tags = validate_bio_consistency(tokens, tags)

        # 只保留有实体的句子
        if any(t != "O" for t in tags):
            records.append({"tokens": tokens, "tags": tags})

    return records


def main():
    parser = argparse.ArgumentParser(description="生成合成BIO训练数据")
    parser.add_argument("--output", default=None,
                        help="输出JSONL路径 (默认: data/ner_training/augmented_bio_data.jsonl)")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="生成样本数 (默认: 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    output_path = args.output or os.path.join(config.NER_TRAINING_DIR, "augmented_bio_data.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = generate_synthetic_data(args.n_samples)
    random.shuffle(records)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"生成 {len(records)} 条合成BIO标注数据 → {output_path}")

    # 标签分布
    tag_counts = {}
    for rec in records:
        for tag in rec["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print("\n标签分布:")
    for tag in sorted(tag_counts.keys()):
        print(f"  {tag}: {tag_counts[tag]}")


if __name__ == "__main__":
    main()
