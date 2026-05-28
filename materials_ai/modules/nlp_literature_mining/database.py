"""数据存储层 — MongoDB主 + JSON回退, 管理论文/三元组的CRUD"""

import os
import json
import hashlib
from typing import List, Dict, Optional
from dataclasses import asdict


class TripletStore:
    """材料-性能-数值 三元组存储. MongoDB优先, 自动回退JSON."""

    def __init__(self, use_mongodb: bool = None, mongodb_uri: str = None,
                 json_path: str = None):
        if json_path is None:
            from config import TRIPLETS_FILE
            json_path = TRIPLETS_FILE
        if mongodb_uri is None:
            from config import MONGODB_URI
            mongodb_uri = mongodb_uri
        self._json_path = json_path
        self._mongodb_uri = mongodb_uri
        self._backend = None
        self._client = None
        self._collection = None
        if use_mongodb is None:
            use_mongodb = self._try_mongodb()
        self._backend = "mongodb" if use_mongodb else "json"

    def _try_mongodb(self) -> bool:
        try:
            from pymongo import MongoClient
            from config import MONGODB_DB, MONGODB_COLLECTION
            self._client = MongoClient(self._mongodb_uri, serverSelectionTimeoutMS=2000)
            self._client.admin.command('ping')
            db = self._client[MONGODB_DB]
            self._collection = db[MONGODB_COLLECTION]
            return True
        except Exception:
            return False

    @property
    def backend(self) -> str:
        return self._backend

    def insert_paper(self, paper_data: dict) -> str:
        """插入解析后的论文, 返回paper_id."""
        paper_id = paper_data.get("paper_id", self._make_id(paper_data.get("filename", "")))
        paper_data["paper_id"] = paper_id
        if self._backend == "mongodb":
            self._collection.update_one(
                {"paper_id": paper_id}, {"$set": paper_data}, upsert=True
            )
        else:
            papers = self._json_load()
            for i, p in enumerate(papers):
                if p.get("paper_id") == paper_id:
                    papers[i] = paper_data
                    break
            else:
                papers.append(paper_data)
            self._json_save(papers)
        return paper_id

    def insert_triplets(self, paper_id: str, triplets: List[dict]):
        """批量插入三元组. 每个triplet: {material, property, value, evidence, confidence}."""
        for t in triplets:
            t["paper_id"] = paper_id
            t["triplet_id"] = self._make_id(
                f"{t.get('material','')}{t.get('property','')}{t.get('value','')}"
            )
        if self._backend == "mongodb":
            self._collection.insert_many(triplets)
        else:
            triplets_data = self._json_load_triplets()
            triplets_data.extend(triplets)
            self._json_save_triplets(triplets_data)

    def query_triplets(self, material: str = None, property: str = None,
                       value_min: float = None, value_max: float = None,
                       year_from: int = None, limit: int = 100) -> List[dict]:
        """多条件查询三元组."""
        if self._backend == "mongodb":
            filt = {}
            if material:
                filt["material"] = {"$regex": material, "$options": "i"}
            if property:
                filt["property"] = {"$regex": property, "$options": "i"}
            if value_min is not None or value_max is not None:
                val_filt = {}
                if value_min is not None:
                    val_filt["$gte"] = value_min
                if value_max is not None:
                    val_filt["$lte"] = value_max
                if val_filt:
                    filt["value_numeric"] = val_filt
            return list(self._collection.find(filt).limit(limit))
        else:
            results = self._json_load_triplets()
            if material:
                results = [r for r in results if material.lower() in r.get("material", "").lower()]
            if property:
                results = [r for r in results if property.lower() in r.get("property", "").lower()]
            if value_min is not None or value_max is not None:
                filtered = []
                for r in results:
                    try:
                        v = float(r.get("value", r.get("value_numeric", 0)))
                    except (ValueError, TypeError):
                        continue
                    if (value_min is None or v >= value_min) and (value_max is None or v <= value_max):
                        filtered.append(r)
                results = filtered
            return results[:limit]

    def query_by_material(self, material_name: str) -> List[dict]:
        """查询某材料的所有三元组."""
        return self.query_triplets(material=material_name)

    def get_statistics(self) -> dict:
        """聚合统计: 论文数/三元组数/唯一材料/属性."""
        if self._backend == "mongodb":
            pipeline = [
                {"$group": {"_id": None, "total": {"$sum": 1},
                 "unique_materials": {"$addToSet": "$material"},
                 "unique_properties": {"$addToSet": "$property"}}}
            ]
            agg = list(self._collection.aggregate(pipeline))
            if agg:
                return {
                    "total_triplets": agg[0]["total"],
                    "unique_materials": len(agg[0]["unique_materials"]),
                    "unique_properties": len(agg[0]["unique_properties"]),
                }
            return {"total_triplets": 0}
        else:
            data = self._json_load_triplets()
            mat = set(r.get("material", "") for r in data)
            prop = set(r.get("property", "") for r in data)
            return {
                "total_triplets": len(data),
                "unique_materials": len(mat),
                "unique_properties": len(prop),
            }

    def _json_load(self) -> List[dict]:
        return _json_read(self._json_path)

    def _json_save(self, data: List[dict]):
        _json_write(self._json_path, data)

    def _json_load_triplets(self) -> List[dict]:
        tp = self._json_path.replace(".json", "_triplets.json")
        return _json_read(tp)

    def _json_save_triplets(self, data: List[dict]):
        tp = self._json_path.replace(".json", "_triplets.json")
        _json_write(tp, data)

    @staticmethod
    def _make_id(s: str) -> str:
        return hashlib.md5(s.encode()).hexdigest()[:12]


class PaperStore:
    """论文全文存储. MongoDB优先, 自动回退JSON (复用现有parsed_papers.json)."""

    def __init__(self, use_mongodb: bool = None, mongodb_uri: str = None,
                 json_path: str = None):
        if json_path is None:
            from config import PARSED_PAPERS_FILE
            json_path = PARSED_PAPERS_FILE
        if mongodb_uri is None:
            from config import MONGODB_URI
            mongodb_uri = mongodb_uri
        self._json_path = json_path
        self._backend = None
        self._client = None
        self._collection = None
        if use_mongodb is None:
            use_mongodb = self._try_mongodb(mongodb_uri)
        self._backend = "mongodb" if use_mongodb else "json"

    def _try_mongodb(self, uri: str) -> bool:
        try:
            from pymongo import MongoClient
            from config import MONGODB_DB
            self._client = MongoClient(uri, serverSelectionTimeoutMS=2000)
            self._client.admin.command('ping')
            self._collection = self._client[MONGODB_DB]["papers"]
            return True
        except Exception:
            return False

    @property
    def backend(self) -> str:
        return self._backend

    def insert(self, paper: dict) -> str:
        """插入或更新论文, 返回paper_id."""
        pid = paper.get("paper_id", TripletStore._make_id(paper.get("filename", "")))
        paper["paper_id"] = pid
        if self._backend == "mongodb":
            self._collection.update_one({"paper_id": pid}, {"$set": paper}, upsert=True)
        else:
            papers = self._json_load()
            for i, p in enumerate(papers):
                if p.get("paper_id") == pid:
                    papers[i] = paper
                    break
            else:
                papers.append(paper)
            self._json_save(papers)
        return pid

    def get(self, paper_id: str) -> Optional[dict]:
        """按ID获取论文."""
        if self._backend == "mongodb":
            return self._collection.find_one({"paper_id": paper_id})
        for p in self._json_load():
            if p.get("paper_id") == paper_id:
                return p
        return None

    def list_all(self) -> List[dict]:
        """列出所有论文."""
        if self._backend == "mongodb":
            return list(self._collection.find({}))
        return self._json_load()

    def count(self) -> int:
        if self._backend == "mongodb":
            return self._collection.count_documents({})
        return len(self._json_load())

    def search_by_keyword(self, keyword: str, limit: int = 20) -> List[dict]:
        """关键词搜索论文全文."""
        kw = keyword.lower()
        if self._backend == "mongodb":
            return list(self._collection.find(
                {"$text": {"$search": keyword}}
            ).limit(limit))
        results = []
        for p in self._json_load():
            if kw in p.get("raw_text", "").lower() or kw in p.get("abstract", "").lower():
                results.append(p)
                if len(results) >= limit:
                    break
        return results

    def _json_load(self) -> List[dict]:
        return _json_read(self._json_path)

    def _json_save(self, data: List[dict]):
        _json_write(self._json_path, data)


def _json_read(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _json_write(path: str, data: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
