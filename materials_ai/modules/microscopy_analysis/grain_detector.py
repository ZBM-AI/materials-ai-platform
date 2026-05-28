"""晶粒实例分割检测 + 截线法晶粒尺寸测量"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .preprocessing import MicrographPreprocessor


class GrainDetector:
    """晶粒检测与尺寸测量.

    支持两种模式:
    - YOLOv8-seg: 深度学习实例分割 (需要ultralytics)
    - 传统分水岭: OpenCV Watershed + 连通域分析 (始终可用)
    """

    def __init__(self, model_path: str = None, device: str = "cpu"):
        self.model = None
        self.model_path = model_path
        self.device = device
        self.preprocessor = MicrographPreprocessor()
        self._model_loaded = False

        if model_path and os.path.exists(model_path):
            self._load_yolo(model_path)

    def _load_yolo(self, model_path: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self._model_loaded = True
        except ImportError:
            pass  # YOLOv8 not installed, fall back to watershed

    @property
    def use_deep_learning(self) -> bool:
        return self._model_loaded

    def detect_grains(self, image: np.ndarray,
                      pixel_scale_um: float = None,
                      min_grain_area_px: int = 100
                      ) -> Dict:
        """检测晶粒并测量尺寸.

        Args:
            image: BGR/Grayscale显微图像
            pixel_scale_um: 标尺 (μm/pixel). 若不提供则仅返回像素单位
            min_grain_area_px: 最小晶粒面积 (过滤噪点)

        Returns:
            {
                "num_grains": int,
                "grain_contours": [(N, 2) array],  # 晶粒轮廓列表
                "grain_areas": [float],             # 每晶粒面积 (μm²或px²)
                "grain_diameters": [float],          # 每晶粒等效直径
                "avg_grain_size_um": float,          # 平均晶粒尺寸
                "grain_size_astm": float,            # ASTM晶粒度
                "intercept_length_um": float,        # 截线法平均截距
                "annotated_image": ndarray,          # 标注图像
                "histogram": {bins, counts},         # 晶粒尺寸分布
            }
        """
        if not HAS_CV2:
            raise ImportError("opencv-python not installed")

        gray = self.preprocessor.process(image, denoise=True, equalize=True)

        if self._model_loaded:
            result = self._detect_with_yolo(image, gray)
        else:
            result = self._detect_with_watershed(gray, min_grain_area_px)

        # 尺寸测量
        if pixel_scale_um is not None:
            result["grain_areas_um2"] = [a * pixel_scale_um ** 2
                                          for a in result["grain_areas_px"]]
            result["grain_diameters_um"] = [
                2.0 * np.sqrt(a / np.pi) * pixel_scale_um
                for a in result["grain_areas_px"]
            ]
            result["avg_grain_size_um"] = float(np.mean(result["grain_diameters_um"])) if result["grain_diameters_um"] else 0.0

            # ASTM 晶粒度: G = -6.644 * log10(D) - 3.288  (D in mm)
            avg_mm = result["avg_grain_size_um"] / 1000.0 if result["avg_grain_size_um"] > 0 else 0
            if avg_mm > 0:
                result["grain_size_astm"] = round(-6.644 * np.log10(avg_mm) - 3.288, 1)
            else:
                result["grain_size_astm"] = 0

            # 截线法
            intercept = self._intercept_method(gray, pixel_scale_um)
            result["intercept_length_um"] = intercept
        else:
            result["avg_grain_size_um"] = 0.0
            result["grain_size_astm"] = 0.0
            result["intercept_length_um"] = 0.0

        return result

    def _detect_with_watershed(self, gray: np.ndarray,
                               min_area: int = 100) -> Dict:
        """分水岭 + 连通域分析检测晶粒."""
        # Otsu + 形态学
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 距离变换 + 分水岭
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # 标记
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        markers[unknown == 255] = 0

        if gray.ndim == 2:
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            color = gray.copy()
        markers = cv2.watershed(color, markers.copy())

        # 提取每个晶粒的轮廓和面积
        contours_data = []
        grain_areas = []
        n_labels = markers.max()

        for label in range(2, n_labels + 1):
            grain_mask = (markers == label).astype(np.uint8)
            area = int(grain_mask.sum())
            if area < min_area:
                continue

            ctrs, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            if ctrs:
                contours_data.append(ctrs[0])
                grain_areas.append(area)

        # 标注图像
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for ctr in contours_data:
            cv2.drawContours(annotated, [ctr], -1, (0, 255, 0), 1)

        # 尺寸直方图
        hist = np.histogram(grain_areas, bins=min(20, max(5, len(grain_areas) // 5)))

        return {
            "num_grains": len(grain_areas),
            "grain_contours": contours_data,
            "grain_areas_px": grain_areas,
            "annotated_image": annotated,
            "histogram_bins": hist[1].tolist(),
            "histogram_counts": hist[0].tolist(),
            "method": "watershed",
        }

    def _detect_with_yolo(self, image: np.ndarray, gray: np.ndarray) -> Dict:
        """YOLOv8-seg实例分割检测晶粒."""
        results = self.model(image, device=self.device, verbose=False)

        contours_data = []
        grain_areas = []

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            h, w = gray.shape[:2]

            for mask in masks:
                mask_resized = cv2.resize(mask.astype(np.float32), (w, h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                area = int(mask_binary.sum())
                ctrs, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
                if ctrs:
                    contours_data.append(ctrs[0])
                    grain_areas.append(area)

        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for ctr in contours_data:
            cv2.drawContours(annotated, [ctr], -1, (0, 255, 0), 1)

        hist = np.histogram(grain_areas, bins=min(20, max(5, len(grain_areas) // 5)))

        return {
            "num_grains": len(grain_areas),
            "grain_contours": contours_data,
            "grain_areas_px": grain_areas,
            "annotated_image": annotated,
            "histogram_bins": hist[1].tolist(),
            "histogram_counts": hist[0].tolist(),
            "method": "yolov8_seg",
        }

    def _intercept_method(self, gray: np.ndarray,
                          pixel_scale_um: float,
                          n_lines: int = 10) -> float:
        """截线法 (Heyn intercept method) 测量平均晶粒尺寸.

        画N条等距水平线, 统计每条线与晶界的交点数,
        截线长度 = 总线长 / 总交点数.

        Returns:
            平均截距 (μm)
        """
        h, w = gray.shape[:2]
        edges = cv2.Canny(gray, 50, 150)
        total_length = 0.0
        total_intersections = 0

        for i in range(1, n_lines + 1):
            y = int(h * i / (n_lines + 1))
            line = edges[y, :]
            # 统计边沿跳变 (0→1) 作为晶界交叉
            transitions = np.diff(line.astype(np.int32))
            n_crossings = int(np.sum(transitions > 0))
            total_intersections += n_crossings
            total_length += w * pixel_scale_um

        if total_intersections == 0:
            return 0.0

        return total_length / total_intersections

    def generate_yolo_training_data(self, labelme_dir: str,
                                     image_dir: str,
                                     output_yaml: str):
        """从Labelme标注生成YOLOv8训练数据配置.

        晶粒标注: polygon标注 → YOLO segmentation格式
        YOLO格式: class_id x1 y1 x2 y2 ... (归一化坐标)
        """
        from .coco_utils import LabelmeToCOCO

        converter = LabelmeToCOCO(categories=["grain"])
        coco = converter.convert_directory(labelme_dir, image_dir)

        # 简化: 直接用COCO格式, 通过ultralytics的COCO支持
        # 或者生成YOLO格式的txt标注文件
        train_dir = os.path.join(os.path.dirname(output_yaml), "yolo_dataset")
        os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)

        yaml_content = f"""
path: {train_dir}
train: images
val: images
nc: 1
names: ['grain']
"""
        with open(output_yaml, 'w') as f:
            f.write(yaml_content.strip())

        return output_yaml


def train_yolo_grain(data_yaml: str, base_model: str = "yolov8n-seg.pt",
                     epochs: int = 100, imgsz: int = 640,
                     device: str = "cpu") -> str:
    """YOLOv8-seg晶粒检测微调.

    Args:
        data_yaml: YOLO格式数据集配置
        base_model: 预训练模型 (yolov8n-seg.pt / yolov8s-seg.pt)
        epochs: 训练轮数
        imgsz: 输入尺寸
        device: 训练设备

    Returns:
        模型保存路径
    """
    try:
        from ultralytics import YOLO
        model = YOLO(base_model)
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            device=device,
            patience=20,
        )
        return str(model.trainer.save_dir)
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")
