"""Labelme→COCO格式转换 + 显微图像数据集工具"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class COCOAnnotation:
    image_id: int
    category_id: int
    bbox: List[float]  # [x, y, width, height]
    segmentation: List[List[float]]  # polygon points
    area: float


class LabelmeToCOCO:
    """将Labelme JSON标注转为COCO格式.

    Labelme格式: {"shapes": [{"label": "...", "points": [[x,y],...], "shape_type": "polygon"}]}
    COCO格式: {"images": [...], "annotations": [...], "categories": [...]}
    """

    def __init__(self, categories: List[str] = None):
        if categories is None:
            categories = [
                "phase_A", "phase_B", "phase_C",  # 物相
                "grain",                            # 晶粒
                "pore", "crack", "inclusion",       # 缺陷
                "martensite", "pearlite", "ferrite", "bainite", "austenite",
            ]
        self.categories = categories
        self.cat_to_id = {c: i + 1 for i, c in enumerate(categories)}
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}

    def convert_file(self, labelme_json_path: str,
                     image_path: str) -> dict:
        """转换单个Labelme JSON → COCO."""
        with open(labelme_json_path, 'r', encoding='utf-8') as f:
            lm_data = json.load(f)

        if HAS_CV2 and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
            else:
                h, w = lm_data.get("imageHeight", 1024), lm_data.get("imageWidth", 1024)
        else:
            h, w = lm_data.get("imageHeight", 1024), lm_data.get("imageWidth", 1024)

        annotations = []
        for shape in lm_data.get("shapes", []):
            label = shape.get("label", "")
            if label not in self.cat_to_id:
                continue
            points = shape.get("points", [])
            if len(points) < 3:
                continue

            # 多边形 → bbox
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            bbox_w, bbox_h = max_x - min_x, max_y - min_y

            # 多边形面积
            area = self._polygon_area(points)

            annotations.append({
                "category_id": self.cat_to_id[label],
                "bbox": [min_x, min_y, bbox_w, bbox_h],
                "segmentation": [[coord for p in points for coord in p]],
                "area": area,
            })

        return {
            "image": {
                "file_name": os.path.basename(image_path),
                "height": h,
                "width": w,
            },
            "annotations": annotations,
        }

    def convert_directory(self, labelme_dir: str, image_dir: str,
                          output_path: str = None) -> dict:
        """批量转换目录中所有Labelme标注.

        Returns:
            COCO格式dict: {"images": [...], "annotations": [...], "categories": [...]}
        """
        coco = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": cid, "name": cname, "supercategory": "microstructure"}
                for cname, cid in self.cat_to_id.items()
            ],
        }

        anno_id = 0
        for i, fname in enumerate(sorted(os.listdir(labelme_dir))):
            if not fname.endswith('.json'):
                continue

            lm_path = os.path.join(labelme_dir, fname)
            # 匹配图像 (同名的.png/.jpg/.tif)
            base = os.path.splitext(fname)[0]
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                candidate = os.path.join(image_dir, base + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path is None:
                img_path = os.path.join(image_dir, base + '.png')

            try:
                result = self.convert_file(lm_path, img_path)
            except Exception:
                continue

            img_info = result["image"]
            img_info["id"] = i
            coco["images"].append(img_info)

            for ann in result["annotations"]:
                ann["id"] = anno_id
                ann["image_id"] = i
                coco["annotations"].append(ann)
                anno_id += 1

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(coco, f, ensure_ascii=False, indent=2)

        return coco

    def coco_to_masks(self, coco_data: dict, image_size: Tuple[int, int]
                      ) -> Dict[int, np.ndarray]:
        """从COCO标注生成逐像素mask (每类别一张)."""
        masks = {}
        for cat_id in self.id_to_cat:
            masks[cat_id] = np.zeros(image_size, dtype=np.uint8)

        for ann in coco_data.get("annotations", []):
            cat_id = ann.get("category_id", 0)
            if cat_id not in masks:
                continue
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(masks[cat_id], [pts], 1)

        return masks

    @staticmethod
    def _polygon_area(points: List[List[float]]) -> float:
        n = len(points)
        area = 0.0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0


class MicrographDataset:
    """显微图像数据集加载器 (支持COCO/JSON)."""

    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (512, 512)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.samples = []

    def load_coco(self, coco_json_path: str) -> List[dict]:
        """加载COCO格式数据集, 返回 [(image, mask_dict), ...]."""
        with open(coco_json_path, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        converter = LabelmeToCOCO()
        samples = []

        for img_info in coco.get("images", []):
            img_path = os.path.join(os.path.dirname(coco_json_path),
                                    img_info.get("file_name", ""))
            if not os.path.exists(img_path):
                img_path = os.path.join(self.data_dir, img_info.get("file_name", ""))

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, self.image_size)

            # 构建此图像的mask
            img_id = img_info["id"]
            masks = {}
            for ann in coco.get("annotations", []):
                if ann.get("image_id") != img_id:
                    continue
                cat_id = ann.get("category_id")
                masks.setdefault(cat_id, np.zeros(self.image_size, dtype=np.uint8))
                seg = ann.get("segmentation", [])
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    # 缩放坐标
                    orig_h, orig_w = img_info["height"], img_info["width"]
                    scale_x = self.image_size[1] / orig_w
                    scale_y = self.image_size[0] / orig_h
                    pts[:, 0] = pts[:, 0] * scale_x
                    pts[:, 1] = pts[:, 1] * scale_y
                    pts = pts.astype(np.int32)
                    cv2.fillPoly(masks[cat_id], [pts], 1)

            samples.append({"image": img, "masks": masks, "file_name": img_info.get("file_name", "")})

        self.samples = samples
        return samples

    @staticmethod
    def binary_mask(masks: Dict[int, np.ndarray],
                    cat_ids: List[int]) -> np.ndarray:
        """合并多个类别的mask为二值mask."""
        result = np.zeros_like(list(masks.values())[0]) if masks else np.zeros((512, 512), dtype=np.uint8)
        for cid in cat_ids:
            if cid in masks:
                result = np.maximum(result, masks[cid])
        return result
