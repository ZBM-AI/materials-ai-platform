"""缺陷检测与分类 — 孔洞/裂纹/夹杂物 + 定量统计"""

import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .preprocessing import MicrographPreprocessor


class DefectAnalyzer:
    """缺陷识别与分类.

    缺陷类型: pore(孔洞), crack(裂纹), inclusion(夹杂物)
    方法: 传统CV形态特征分类 + (可选)CNN分类器
    """

    DEFECT_TYPES = ["pore", "crack", "inclusion"]

    def __init__(self, cnn_model_path: str = None):
        self.preprocessor = MicrographPreprocessor()
        self.cnn_model = None
        if cnn_model_path and os.path.exists(cnn_model_path):
            self._load_cnn(cnn_model_path)

    def _load_cnn(self, path: str):
        try:
            import torch
            self.cnn_model = torch.load(path, map_location="cpu")
        except Exception:
            pass

    def analyze(self, image: np.ndarray,
                min_defect_area_px: int = 20) -> Dict:
        """检测并分类缺陷.

        Returns:
            {
                "total_defects": int,
                "pores": [{"area":, "perimeter":, "circularity":, "bbox":, "centroid":}],
                "cracks": [...],
                "inclusions": [...],
                "defect_fraction": float,  # 缺陷面积占比
                "annotated_image": ndarray,
            }
        """
        if not HAS_CV2:
            raise ImportError("opencv-python not installed")

        gray = self.preprocessor.process(image, denoise=True, equalize=False)

        # 自适应阈值检测异常区域
        binary = self._detect_anomalies(gray)

        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        defects = {"pores": [], "cracks": [], "inclusions": [],
                    "total_defects": 0, "defect_fraction": 0.0,
                    "annotated_image": None}

        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        total_defect_px = 0

        # 颜色: 孔洞红, 裂纹蓝, 夹杂物黄
        color_map = {
            "pore": (0, 0, 255),
            "crack": (255, 0, 0),
            "inclusion": (0, 255, 255),
        }

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < min_defect_area_px:
                continue

            x = stats[label_idx, cv2.CC_STAT_LEFT]
            y = stats[label_idx, cv2.CC_STAT_TOP]
            w = stats[label_idx, cv2.CC_STAT_WIDTH]
            h = stats[label_idx, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[label_idx]

            # 提取该区域的mask
            region_mask = (labels == label_idx).astype(np.uint8) * 255

            # 形态特征
            region_gray = gray[y:y + h, x:x + w]
            features = self._extract_features(region_mask[y:y + h, x:x + w],
                                               area, w, h)

            # 分类
            defect_type = self._classify_defect(features)

            defect_info = {
                "area_px": int(area),
                "perimeter": float(features["perimeter"]),
                "circularity": float(features["circularity"]),
                "aspect_ratio": float(features["aspect_ratio"]),
                "bbox": [int(x), int(y), int(w), int(h)],
                "centroid": [float(cx), float(cy)],
                "mean_intensity": float(features["mean_intensity"]),
            }

            defects[defect_type + "s"].append(defect_info)
            total_defect_px += area

            # 绘制
            color = color_map.get(defect_type, (128, 128, 128))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, defect_type[:4], (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        defects["total_defects"] = sum(
            len(defects[t + "s"]) for t in self.DEFECT_TYPES
        )
        defects["defect_fraction"] = total_defect_px / (gray.shape[0] * gray.shape[1])
        defects["annotated_image"] = annotated

        return defects

    def _detect_anomalies(self, gray: np.ndarray) -> np.ndarray:
        """检测异常区域 (暗/亮缺陷)."""
        # 暗缺陷 (孔洞、裂纹)
        _, dark = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 亮缺陷 (夹杂物)
        _, bright = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学清理
        kernel = np.ones((2, 2), np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

        return cv2.bitwise_or(dark, bright)

    def _extract_features(self, region: np.ndarray, area: float,
                          width: int, height: int) -> dict:
        """提取缺陷区域形态特征."""
        ctrs, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        if not ctrs:
            return {"perimeter": 0, "circularity": 1, "aspect_ratio": 1,
                    "mean_intensity": 0}

        cnt = max(ctrs, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        # 圆形度: 4πA/P²  (圆=1, 越扁/不规则越接近0)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        mean_intensity = region.mean() if region.any() else 0

        return {
            "perimeter": perimeter,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "mean_intensity": mean_intensity,
        }

    def _classify_defect(self, features: dict) -> str:
        """基于形态特征分类缺陷类型.

        规则:
        - 孔洞: 高圆形度 (>0.7), 低长宽比 (<2)
        - 裂纹: 低圆形度 (<0.4), 高长宽比 (>3)
        - 夹杂物: 中等圆形度, 低长宽比 (<3), 高/低强度
        """
        circ = features["circularity"]
        ar = features["aspect_ratio"]

        if circ > 0.7 and ar < 2.0:
            return "pore"
        elif circ < 0.4 or ar > 3.0:
            return "crack"
        else:
            return "inclusion"

    def defect_summary_table(self, analysis: Dict,
                             pixel_scale_um: float = None) -> List[dict]:
        """生成缺陷汇总表."""
        rows = []
        for dtype in self.DEFECT_TYPES:
            items = analysis.get(dtype + "s", [])
            if not items:
                continue
            areas_px = [d["area_px"] for d in items]
            row = {
                "type": dtype,
                "count": len(items),
                "total_area_px": sum(areas_px),
                "avg_area_px": np.mean(areas_px),
                "max_area_px": max(areas_px),
                "min_area_px": min(areas_px),
            }
            if pixel_scale_um:
                row["total_area_um2"] = row["total_area_px"] * pixel_scale_um ** 2
                row["avg_area_um2"] = row["avg_area_px"] * pixel_scale_um ** 2
            rows.append(row)
        return rows
