"""显微组织分类器 — 马氏体/珠光体/铁素体/贝氏体/奥氏体"""

import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .preprocessing import MicrographPreprocessor


class MicrostructureClassifier:
    """显微组织分类.

    支持:
    - 传统纹理特征 (GLCM + LBP) + 分类器
    - (可选) CNN分类器
    """

    STRUCTURES = [
        "martensite",   # 马氏体: 针状/板条, 高硬度
        "pearlite",     # 珠光体: 层片状, 铁素体+渗碳体
        "ferrite",      # 铁素体: 等轴晶
        "bainite",      # 贝氏体: 羽毛状/针状
        "austenite",    # 奥氏体: 等轴晶+孪晶
        "spheroidite",  # 球化体: 球状碳化物
    ]

    def __init__(self, model_path: str = None):
        self.preprocessor = MicrographPreprocessor()
        self.model = None
        self.model_path = model_path
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        try:
            import torch
            self.model = torch.load(path, map_location="cpu")
        except Exception:
            pass

    def classify(self, image: np.ndarray) -> Dict:
        """分类显微组织.

        Returns:
            {
                "predicted_class": str,
                "probabilities": {class_name: prob},
                "features": {feature_name: value},
            }
        """
        if not HAS_CV2:
            raise ImportError("opencv-python not installed")

        gray = self.preprocessor.process(image, denoise=True, equalize=True)

        if self.model is not None:
            return self._classify_with_cnn(gray)

        features = self._extract_texture_features(gray)
        predicted = self._rule_based_classify(features)

        return {
            "predicted_class": predicted,
            "probabilities": self._rule_probs(features),
            "features": {k: float(v) for k, v in features.items()},
            "method": "texture_rules",
        }

    def _extract_texture_features(self, gray: np.ndarray) -> dict:
        """提取纹理特征: GLCM + LBP + 梯度统计."""
        # LBP (Local Binary Pattern) 直方图均值
        lbp = self._lbp(gray)
        lbp_mean = float(lbp.mean())
        lbp_std = float(lbp.std())

        # GLCM (Gray-Level Co-occurrence Matrix)
        glcm = self._glcm_features(gray)

        # 梯度统计
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # 强度统计
        return {
            "lbp_mean": lbp_mean,
            "lbp_std": lbp_std,
            "glcm_contrast": glcm.get("contrast", 0),
            "glcm_homogeneity": glcm.get("homogeneity", 0),
            "glcm_correlation": glcm.get("correlation", 0),
            "glcm_energy": glcm.get("energy", 0),
            "gradient_mean": float(grad_mag.mean()),
            "gradient_std": float(grad_mag.std()),
            "intensity_mean": float(gray.mean()),
            "intensity_std": float(gray.std()),
            "entropy": float(self._entropy(gray)),
        }

    def _lbp(self, gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """简化LBP特征图."""
        h, w = gray.shape
        lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(j + radius * np.cos(angle))
                    y = int(i - radius * np.sin(angle))
                    if 0 <= x < w and 0 <= y < h:
                        if gray[y, x] >= center:
                            code |= 1 << k
                lbp[i - radius, j - radius] = code
        return lbp

    def _glcm_features(self, gray: np.ndarray, distances: List[int] = None,
                       angles: List[float] = None, levels: int = 64) -> dict:
        """计算GLCM纹理特征."""
        if distances is None:
            distances = [1]
        if angles is None:
            angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        # 量化到levels级
        if gray.max() > 0:
            gray_q = ((gray.astype(np.float32) / gray.max()) * (levels - 1)).astype(np.uint8)
        else:
            gray_q = gray.astype(np.uint8)

        glcm = np.zeros((levels, levels), dtype=np.float64)

        for d in distances:
            for angle in angles:
                dx = int(d * np.cos(angle))
                dy = int(d * np.sin(angle))
                h, w = gray_q.shape
                for i in range(max(0, -dy), min(h, h - dy)):
                    for j in range(max(0, -dx), min(w, w - dx)):
                        glcm[gray_q[i, j], gray_q[i + dy, j + dx]] += 1

        glcm /= glcm.sum() + 1e-10

        i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')
        contrast = np.sum(((i - j) ** 2) * glcm)
        homogeneity = np.sum(glcm / (1 + (i - j) ** 2))
        energy = np.sum(glcm ** 2)
        mu_i = np.sum(i * glcm)
        mu_j = np.sum(j * glcm)
        sigma_i = np.sqrt(np.sum((i - mu_i) ** 2 * glcm))
        sigma_j = np.sqrt(np.sum((j - mu_j) ** 2 * glcm))
        correlation = np.sum((i - mu_i) * (j - mu_j) * glcm) / (sigma_i * sigma_j + 1e-10)

        return {
            "contrast": float(contrast),
            "homogeneity": float(homogeneity),
            "energy": float(energy),
            "correlation": float(correlation),
        }

    def _entropy(self, gray: np.ndarray) -> float:
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        hist = hist / (hist.sum() + 1e-10)
        return -np.sum(hist * np.log2(hist + 1e-10))

    def _rule_based_classify(self, f: dict) -> str:
        """基于纹理特征规则的组织分类.

        - 马氏体: 高梯度, 高LBP标准差 (针状/板条导致方向性纹理)
        - 珠光体: 高LBP均值, 中等对比度 (层片状)
        - 铁素体: 低梯度, 低对比度 (均匀等轴晶)
        - 贝氏体: 中高梯度, 较高熵 (比马氏体粗, 比珠光体细)
        """
        scores = {}

        # 马氏体: 高梯度_std + 低glcm_homogeneity
        scores["martensite"] = (
            f["gradient_std"] / 50.0 +
            (1.0 - f["glcm_homogeneity"]) * 2.0 +
            f["lbp_std"] / 30.0
        )
        # 珠光体: 高glcm_contrast + 中梯度
        scores["pearlite"] = (
            f["glcm_contrast"] / 200.0 +
            f["lbp_mean"] / 80.0 +
            (1.0 - f["gradient_std"] / 60.0)
        )
        # 铁素体: 低Gradient + 高Homogeneity
        scores["ferrite"] = (
            (1.0 - f["gradient_std"] / 40.0) * 2.0 +
            f["glcm_homogeneity"] * 2.0 +
            (1.0 - f["glcm_contrast"] / 250.0)
        )
        # 贝氏体: 中等特征
        scores["bainite"] = (
            f["glcm_contrast"] / 150.0 +
            f["gradient_std"] / 40.0 +
            f["entropy"] / 7.0
        )
        # 奥氏体: 低梯度 + 孪晶特征 (中LBP)
        scores["austenite"] = (
            (1.0 - f["gradient_std"] / 35.0) * 1.5 +
            f["lbp_std"] / 25.0 +
            f["glcm_homogeneity"] * 1.5
        )

        return max(scores, key=scores.get)

    def _rule_probs(self, f: dict) -> Dict[str, float]:
        """将规则分数转为近似概率."""
        scores = {}
        scores["martensite"] = f["gradient_std"] / 50.0 + (1.0 - f["glcm_homogeneity"]) * 2.0 + f["lbp_std"] / 30.0
        scores["pearlite"] = f["glcm_contrast"] / 200.0 + f["lbp_mean"] / 80.0 + (1.0 - f["gradient_std"] / 60.0)
        scores["ferrite"] = (1.0 - f["gradient_std"] / 40.0) * 2.0 + f["glcm_homogeneity"] * 2.0 + (1.0 - f["glcm_contrast"] / 250.0)
        scores["bainite"] = f["glcm_contrast"] / 150.0 + f["gradient_std"] / 40.0 + f["entropy"] / 7.0
        scores["austenite"] = (1.0 - f["gradient_std"] / 35.0) * 1.5 + f["lbp_std"] / 25.0 + f["glcm_homogeneity"] * 1.5
        total = sum(scores.values())
        return {k: v / total if total > 0 else 0.2 for k, v in scores.items()}

    def _classify_with_cnn(self, gray: np.ndarray) -> Dict:
        """CNN分类器 (如已加载)."""
        try:
            import torch
            img_t = torch.tensor(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(img_t)
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()
            pred_idx = int(np.argmax(probs))
            return {
                "predicted_class": self.STRUCTURES[pred_idx] if pred_idx < len(self.STRUCTURES) else "unknown",
                "probabilities": {self.STRUCTURES[i]: float(p) for i, p in enumerate(probs)},
                "method": "cnn",
            }
        except Exception:
            return {"predicted_class": "unknown", "probabilities": {}, "method": "cnn_error"}
