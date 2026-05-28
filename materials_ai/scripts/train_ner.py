"""独立SciBERT微调脚本 — 命令行训练NER模型.

用法:
    python scripts/train_ner.py --data data/ner_training/
    python scripts/train_ner.py --data data/ner_training/ --epochs 5 --batch-size 16
"""

import os
import sys
import json
import argparse
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.nlp_literature_mining.ner_trainer import SciBERTNERTrainer
import config


def load_jsonl_files(data_path: str) -> list:
    """加载一个目录或单个JSONL文件中的所有训练数据."""
    if os.path.isdir(data_path):
        files = glob.glob(os.path.join(data_path, "*.jsonl"))
    else:
        files = [data_path]

    all_data = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_data.append(json.loads(line))
    return all_data


def main():
    parser = argparse.ArgumentParser(description="微调SciBERT进行材料科学NER")
    parser.add_argument("--data", required=True,
                        help="训练数据目录或JSONL文件路径")
    parser.add_argument("--model-name", default=None,
                        help=f"预训练模型名 (默认: {config.SCIBERT_MODEL_NAME})")
    parser.add_argument("--output-dir", default=None,
                        help=f"模型输出目录 (默认: {config.NER_MODEL_DIR})")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=None, help="最大序列长度")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="验证集比例 (默认: 0.15)")
    parser.add_argument("--no-save", action="store_true", help="不保存模型")
    args = parser.parse_args()

    model_name = args.model_name or config.SCIBERT_MODEL_NAME
    output_dir = args.output_dir or config.NER_MODEL_DIR
    epochs = args.epochs or config.SCIBERT_NUM_EPOCHS
    batch_size = args.batch_size or config.SCIBERT_BATCH_SIZE
    lr = args.lr or config.SCIBERT_LEARNING_RATE
    max_length = args.max_length or config.SCIBERT_MAX_LENGTH

    # 加载数据
    print(f"从 {args.data} 加载训练数据...")
    data = load_jsonl_files(args.data)
    if not data:
        print("错误: 未找到训练数据。请先运行 generate_seed_bio_data.py 和 augment_ner_data.py")
        sys.exit(1)

    print(f"已加载 {len(data)} 条标注数据")

    # 标签分布
    tag_counts = {}
    for item in data:
        for tag in item.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print("训练数据标签分布:")
    for tag in sorted(tag_counts.keys()):
        print(f"  {tag}: {tag_counts[tag]}")

    # 训练
    print(f"\n开始训练 SciBERT NER 模型...")
    print(f"  模型: {model_name}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  输出: {output_dir}")

    trainer = SciBERTNERTrainer(
        model_name=model_name,
        output_dir=output_dir,
        max_length=max_length,
        batch_size=batch_size,
        learning_rate=lr,
        num_epochs=epochs,
        tag_to_id=config.TAG_TO_ID,
    )

    trainer.train(train_data=data, val_split=args.val_split, save_best=not args.no_save)

    # 评估
    print("\n评估模型...")
    split = max(1, int(len(data) * args.val_split))
    test_data = data[-split:] if split > 0 else data[-max(1, len(data) // 10):]
    metrics = trainer.evaluate(test_data)
    if isinstance(metrics, dict):
        print("\nClassification Report:")
        for label, vals in metrics.items():
            if isinstance(vals, dict):
                f1 = vals.get("f1-score", 0)
                prec = vals.get("precision", 0)
                rec = vals.get("recall", 0)
                sup = vals.get("support", 0)
                if sup > 0:
                    print(f"  {label:15s}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  (n={sup})")

    print(f"\n模型已保存到: {output_dir}")


if __name__ == "__main__":
    main()
