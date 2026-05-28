"""物相分割推理 + 相比例统计"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .unet_model import UNet
from .preprocessing import MicrographPreprocessor

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PhaseSegmenter:
    """物相分割推理引擎.

    加载已训练的U-Net模型, 对显微图像进行物相分割并统计各相面积比例.
    如果模型不可用, 自动回退到传统方法 (Otsu阈值 + Watershed).
    """

    def __init__(self, model_path: str = None, n_classes: int = 4,
                 class_names: List[str] = None, device: str = None):
        self.n_classes = n_classes
        self.class_names = class_names or [
            "background", "phase_A", "phase_B", "phase_C"
        ]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocessor = MicrographPreprocessor()

        if model_path and os.path.exists(model_path) and HAS_TORCH:
            self.model = UNet(n_channels=1, n_classes=n_classes)
            ckpt = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.n_classes = ckpt.get("n_classes", n_classes)

    def segment(self, image: np.ndarray) -> Dict:
        """物相分割.

        Args:
            image: BGR/Grayscale显微图像

        Returns:
            {
                "segmentation_map": (H, W) class index array,
                "confidence_map": (H, W) probability map,
                "phase_fractions": {class_name: area_fraction},
                "phase_areas_um2": {class_name: area_in_square_um},
                "method": "unet" | "otsu_watershed",
            }
        """
        # 预处理
        gray = self.preprocessor.process(image, denoise=True, equalize=True)

        if self.model is not None:
            return self._segment_with_unet(gray)
        else:
            return self._segment_with_traditional(gray)

    def _segment_with_unet(self, gray: np.ndarray) -> Dict:
        img_t = torch.tensor(
            gray.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_t)
            probs = torch.nn.functional.softmax(logits, dim=1)
            seg_map = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            conf = probs.max(dim=1)[0].squeeze(0).cpu().numpy()

        return self._compute_fractions(seg_map, conf, method="unet")

    def _segment_with_traditional(self, gray: np.ndarray) -> Dict:
        # Otsu + Watershed
        dist, markers = self.preprocessor.watershed_seeds(gray)

        # 在原始图像上用分水岭
        if gray.ndim == 2:
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            color = gray

        markers = cv2.watershed(color, markers.copy())
        seg_map = markers - 1  # 背景= -1 → 0
        seg_map = np.maximum(seg_map, 0)
        seg_map = np.clip(seg_map, 0, self.n_classes - 1)

        conf = np.ones_like(gray, dtype=np.float32) * 0.7
        return self._compute_fractions(seg_map, conf, method="otsu_watershed")

    def _compute_fractions(self, seg_map: np.ndarray,
                           confidence: np.ndarray, method: str) -> Dict:
        total = seg_map.size
        fractions = {}
        for cls_idx in range(self.n_classes):
            mask = seg_map == cls_idx
            area_px = mask.sum()
            name = (self.class_names[cls_idx]
                    if cls_idx < len(self.class_names) else f"class_{cls_idx}")
            fractions[name] = area_px / total

        return {
            "segmentation_map": seg_map,
            "confidence_map": confidence,
            "phase_fractions": fractions,
            "segmented_image": self._colorize(seg_map),
            "method": method,
        }

    def _colorize(self, seg_map: np.ndarray) -> np.ndarray:
        """伪彩色渲染分割结果."""
        colors = [
            (0, 0, 0),       # 0: background - black
            (255, 0, 0),     # 1: phase_A - red
            (0, 255, 0),     # 2: phase_B - green
            (0, 0, 255),     # 3: phase_C - blue
            (255, 255, 0),   # 4: yellow
            (255, 0, 255),   # 5: magenta
            (0, 255, 255),   # 6: cyan
            (128, 128, 128), # 7: gray
        ]
        h, w = seg_map.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_idx in range(self.n_classes):
            if cls_idx < len(colors):
                color_img[seg_map == cls_idx] = colors[cls_idx]
        return color_img

    def compute_phase_statistics(self, seg_map: np.ndarray,
                                 pixel_scale_um: float = None) -> List[dict]:
        """统计各相详细指标.

        Args:
            seg_map: 分割图
            pixel_scale_um: 每像素对应的微米数

        Returns:
            [{phase, area_fraction, num_regions, avg_region_area_um2, ...}, ...]
        """
        stats = []
        for cls_idx in range(1, self.n_classes):
            mask = (seg_map == cls_idx).astype(np.uint8)
            area_frac = mask.sum() / mask.size

            # 连通域分析
            num_labels, labels, areas, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )

            # areas[0] 是背景
            region_areas = areas[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])

            stat = {
                "phase": self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"class_{cls_idx}",
                "area_fraction": area_frac,
                "num_regions": len(region_areas),
                "avg_region_area_px": float(region_areas.mean()) if len(region_areas) > 0 else 0,
                "max_region_area_px": int(region_areas.max()) if len(region_areas) > 0 else 0,
                "min_region_area_px": int(region_areas.min()) if len(region_areas) > 0 else 0,
            }

            if pixel_scale_um is not None:
                stat["avg_region_area_um2"] = stat["avg_region_area_px"] * pixel_scale_um ** 2
                stat["pixel_scale_um_per_px"] = pixel_scale_um

            stats.append(stat)

        return stats
