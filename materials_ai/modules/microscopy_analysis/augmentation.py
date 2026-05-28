"""数据增强策略 — 仿射变换/光照调整/噪声注入 (显微图像专用)"""

import numpy as np
from typing import Tuple, Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class MicrographAugmenter:
    """显微图像增强器.

    支持:
    - 几何变换: 翻转/旋转/缩放/仿射
    - 光照调整: 亮度/对比度/Gamma
    - 噪声注入: 高斯/泊松/椒盐
    - 模糊/锐化
    - 弹性形变 (模拟SEM形变)
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)

    # ---- 几何变换 ----

    def random_flip(self, image: np.ndarray, mask: np.ndarray = None,
                    horizontal: bool = True, vertical: bool = True
                    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.rng.rand() < 0.5 and horizontal:
            image = cv2.flip(image, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        if self.rng.rand() < 0.5 and vertical:
            image = cv2.flip(image, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return image, mask

    def random_rotate(self, image: np.ndarray, mask: np.ndarray = None,
                      angle_range: Tuple[float, float] = (-180, 180)
                      ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        angle = self.rng.uniform(*angle_range)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT)
        return image, mask

    def random_scale(self, image: np.ndarray, mask: np.ndarray = None,
                     scale_range: Tuple[float, float] = (0.8, 1.2)
                     ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        scale = self.rng.uniform(*scale_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 居中裁剪或填充
        if scale > 1.0:
            dh, dw = new_h - h, new_w - w
            image = image[dh // 2:dh // 2 + h, dw // 2:dw // 2 + w]
            if mask is not None:
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                mask = mask[dh // 2:dh // 2 + h, dw // 2:dw // 2 + w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            image = np.pad(image, ((pad_h, h - new_h - pad_h),
                                    (pad_w, w - new_w - pad_w)), mode='reflect')
            if mask is not None:
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                mask = np.pad(mask, ((pad_h, h - new_h - pad_h),
                                      (pad_w, w - new_w - pad_w)), mode='constant')
        return image, mask

    # ---- 光照调整 ----

    def random_brightness_contrast(self, image: np.ndarray,
                                   brightness: float = 0.15,
                                   contrast: float = 0.15
                                   ) -> np.ndarray:
        alpha = 1.0 + self.rng.uniform(-contrast, contrast)
        beta = self.rng.uniform(-brightness * 255, brightness * 255)
        return np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    def random_gamma(self, image: np.ndarray,
                     gamma_range: Tuple[float, float] = (0.7, 1.3)
                     ) -> np.ndarray:
        gamma = self.rng.uniform(*gamma_range)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)

    # ---- 噪声注入 ----

    def random_gaussian_noise(self, image: np.ndarray,
                              sigma_range: Tuple[float, float] = (1, 10)
                              ) -> np.ndarray:
        sigma = self.rng.uniform(*sigma_range)
        noise = self.rng.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def random_salt_pepper(self, image: np.ndarray,
                           prob: float = 0.02) -> np.ndarray:
        img = image.copy()
        salt = self.rng.rand(*image.shape) < prob / 2
        pepper = self.rng.rand(*image.shape) < prob / 2
        img[salt] = 255
        img[pepper] = 0
        return img

    def random_poisson_noise(self, image: np.ndarray) -> np.ndarray:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = self.rng.poisson(image.astype(np.float32) * vals) / float(vals)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    # ---- 弹性形变 ----

    def elastic_deform(self, image: np.ndarray, mask: np.ndarray = None,
                       alpha: float = 30, sigma: float = 4
                       ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """弹性形变 (模拟SEM图像的非刚性畸变)."""
        h, w = image.shape[:2]
        dx = cv2.GaussianBlur(
            self.rng.randn(h, w).astype(np.float32) * alpha, (0, 0), sigma
        )
        dy = cv2.GaussianBlur(
            self.rng.randn(h, w).astype(np.float32) * alpha, (0, 0), sigma
        )
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
        if mask is not None:
            mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_REFLECT)
        return image, mask

    # ---- Pipeline ----

    def apply_pipeline(self, image: np.ndarray, mask: np.ndarray = None,
                       heavy: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """应用随机增强pipeline.

        Args:
            image: 输入图像 (H, W) 或 (H, W, 3)
            mask: 分割mask (H, W)
            heavy: 是否使用强力增强 (弹性形变+多种噪声)
        """
        if not HAS_CV2:
            raise ImportError("opencv-python not installed")

        # 几何
        image, mask = self.random_flip(image, mask)
        image, mask = self.random_rotate(image, mask, angle_range=(-30, 30))
        image, mask = self.random_scale(image, mask, scale_range=(0.85, 1.15))

        # 光照
        image = self.random_brightness_contrast(image)
        image = self.random_gamma(image, gamma_range=(0.8, 1.2))

        # 噪声
        if self.rng.rand() < 0.3:
            image = self.random_gaussian_noise(image, sigma_range=(1, 8))

        if heavy:
            if self.rng.rand() < 0.3:
                image = self.random_salt_pepper(image, prob=0.01)
            if self.rng.rand() < 0.2:
                image, mask = self.elastic_deform(image, mask, alpha=20, sigma=3)

        return image, mask
