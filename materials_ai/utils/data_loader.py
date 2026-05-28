"""数据加载与缓存工具"""
import os
import json
import pandas as pd
import config


class DataLoader:
    @staticmethod
    def load_parsed_papers() -> list:
        if os.path.exists(config.PARSED_PAPERS_FILE):
            with open(config.PARSED_PAPERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    @staticmethod
    def load_seed_kg() -> dict:
        if os.path.exists(config.SEED_KG_FILE):
            with open(config.SEED_KG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"entities": [], "relations": []}

    @staticmethod
    def load_property_data(dataset: str) -> pd.DataFrame:
        paths = {
            "band_gap": config.BAND_GAP_DATA,
            "formation_energy": config.FORMATION_ENERGY_DATA,
            "mechanical": config.MECHANICAL_DATA,
        }
        path = paths.get(dataset)
        if path and os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    @staticmethod
    def save_parsed_papers(papers: list):
        os.makedirs(os.path.dirname(config.PARSED_PAPERS_FILE), exist_ok=True)
        with open(config.PARSED_PAPERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
