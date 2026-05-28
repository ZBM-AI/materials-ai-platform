"""显微图像预处理 — 去噪/光照均衡/超分辨率增强"""

import numpy as np
from typing import Tuple, Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class MicrographPreprocessor:
    """SEM/TEM显微照片预处理pipeline.

    处理流程: 灰度归一化 → 去噪 → CLAHE光照均衡 → (可选)超分辨率
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.clip_limit = clip_limit
        self.tile_grid_size = (tile_grid_size, tile_grid_size)

    def process(self, image: np.ndarray,
                denoise: bool = True,
                equalize: bool = True,
                super_res: bool = False) -> np.ndarray:
        """完整预处理流水线.

        Args:
            image: BGR/Grayscale输入图像 (H, W) 或 (H, W, 3)
            denoise: 是否去噪
            equalize: 是否CLAHE均衡
            super_res: 是否超分辨率增强

        Returns:
            处理后的图像
        """
        if not HAS_CV2:
            raise ImportError("opencv-python not installed")

        img = image.copy()

        # 灰度转换
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 归一化
        gray = self._normalize(gray)

        # 去噪
        if denoise:
            gray = self._denoise(gray)

        # CLAHE 光照均衡
        if equalize:
            gray = self._clahe_equalize(gray)

        # 超分辨率 (简化: 锐化增强)
        if super_res:
            gray = self._enhance_sharpness(gray)

        return gray

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """灰度归一化到 [0, 255]."""
        if img.dtype == np.uint8:
            return img
        img_f = img.astype(np.float32)
        mn, mx = img_f.min(), img_f.max()
        if mx > mn:
            img_f = (img_f - mn) / (mx - mn) * 255.0
        return img_f.astype(np.uint8)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """自适应去噪: 高斯模糊(低噪) + NLM(高噪)."""
        sigma = self._estimate_noise(img)
        if sigma < 15:
            return cv2.GaussianBlur(img, (3, 3), 0)
        elif sigma < 30:
            return cv2.fastNlMeansDenoising(img, None, h=10,
                                             templateWindowSize=7,
                                             searchWindowSize=21)
        else:
            return cv2.fastNlMeansDenoising(img, None, h=sigma * 0.4,
                                             templateWindowSize=7,
                                             searchWindowSize=21)

    def _estimate_noise(self, img: np.ndarray) -> float:
        """估计图像噪声水平 (Laplacian方差法)."""
        lap = cv2.Laplacian(img, cv2.CV_64F)
        return float(lap.std())

    def _clahe_equalize(self, img: np.ndarray) -> np.ndarray:
        """CLAHE 自适应直方图均衡."""
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.tile_grid_size)
        return clahe.apply(img)

    def _enhance_sharpness(self, img: np.ndarray) -> np.ndarray:
        """锐化增强 (边缘保持)."""
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def adaptive_threshold(self, img: np.ndarray,
                           method: str = "otsu") -> np.ndarray:
        """自适应二值化 (用于相分割预处理)."""
        if method == "otsu":
            _, binary = cv2.threshold(img, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(img, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
        else:
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return binary

    def watershed_seeds(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """分水岭预处理: 返回距离图+标记."""
        binary = self.adaptive_threshold(img, "otsu")
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        return dist_transform, markers
