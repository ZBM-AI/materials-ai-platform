"""U-Net语义分割训练pipeline"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple

from .unet_model import UNet, DiceCELoss, compute_iou, compute_dice, mean_iou, pixel_accuracy
from .augmentation import MicrographAugmenter
from .coco_utils import LabelmeToCOCO, MicrographDataset

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class SegmentationDataset(Dataset):
    """PyTorch分割数据集 (从COCO JSON加载)."""

    def __init__(self, coco_json: str, image_size: Tuple[int, int] = (512, 512),
                 n_classes: int = 4, augment: bool = False):
        loader = MicrographDataset(
            os.path.dirname(coco_json) if coco_json else ".",
            image_size,
        )
        self.samples = loader.load_coco(coco_json)
        self.image_size = image_size
        self.n_classes = n_classes
        self.augment = augment
        self.augmenter = MicrographAugmenter() if augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = sample["image"]

        # 归一化到[0, 1]
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # (1, H, W)

        # 构建目标mask (合并所有标注类别)
        masks = sample.get("masks", {})
        target = np.zeros(self.image_size, dtype=np.int64)
        for cat_id, mask in masks.items():
            # cat_id映射到类别索引 (0=bg, 1=phase_A, 2=phase_B, ...)
            cls_idx = min(cat_id, self.n_classes - 1)
            target[mask > 0] = cls_idx

        # 数据增强 (同步变换image和mask)
        if self.augment and self.augmenter is not None:
            img_2d = (image[0] * 255).astype(np.uint8)
            aug_img, aug_mask = self.augmenter.apply_pipeline(img_2d, target.astype(np.uint8))
            if aug_img is not None:
                image = np.expand_dims(aug_img.astype(np.float32) / 255.0, axis=0)
            if aug_mask is not None:
                target = aug_mask.astype(np.int64)

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.long),
        }


class UNetTrainer:
    """U-Net训练器 — 物相分割.

    Args:
        n_classes: 类别数 (含背景)
        n_channels: 输入通道
        base_channels: 基础通道数
        bilinear: 双线性上采样
        lr: 学习率
        device: 训练设备
    """

    def __init__(self, n_classes: int = 4, n_channels: int = 1,
                 base_channels: int = 64, bilinear: bool = True,
                 lr: float = 1e-4, device: str = None):
        if not HAS_TORCH:
            raise ImportError("torch not installed")

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            bilinear=bilinear,
            base_channels=base_channels,
        ).to(self.device)

        self.criterion = DiceCELoss(alpha=0.5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True,
        )

        self.history = {"train_loss": [], "val_loss": [], "val_mIoU": []}

    def train(self, train_dataset: SegmentationDataset,
              val_dataset: SegmentationDataset = None,
              epochs: int = 100, batch_size: int = 8,
              save_dir: str = None) -> dict:
        """训练循环.

        Returns:
            history dict
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=0)

        best_miou = 0.0

        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            self.history["train_loss"].append(avg_loss)

            # Validation
            val_miou = 0.0
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                all_miou = []
                with torch.no_grad():
                    for batch in val_loader:
                        images = batch["image"].to(self.device)
                        targets = batch["target"].to(self.device)
                        outputs = self.model(images)
                        val_loss += self.criterion(outputs, targets).item()
                        all_miou.append(mean_iou(outputs, targets, self.n_classes))

                val_loss /= len(val_loader)
                val_miou = float(np.mean(all_miou))
                self.history["val_loss"].append(val_loss)
                self.history["val_mIoU"].append(val_miou)

                self.scheduler.step(val_loss)

                if val_miou > best_miou and save_dir:
                    best_miou = val_miou
                    self.save(os.path.join(save_dir, "unet_best.pt"))

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Val mIoU: {val_miou:.4f}")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.save(os.path.join(save_dir, "unet_final.pt"))

            # 保存训练历史
            hist_path = os.path.join(save_dir, "training_history.json")
            with open(hist_path, 'w') as f:
                json.dump({k: [float(v) for v in vals]
                           for k, vals in self.history.items()}, f, indent=2)

        return self.history

    def evaluate(self, dataset: SegmentationDataset,
                 class_names: List[str] = None) -> dict:
        """在数据集上评估, 返回每类IoU/Dice."""
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        self.model.eval()
        all_ious = []
        all_dices = []
        total_correct = 0
        total_pixels = 0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device)
                outputs = self.model(images)

                all_ious.append(compute_iou(outputs, targets, self.n_classes))
                all_dices.append(compute_dice(outputs, targets, self.n_classes))
                total_correct += (torch.argmax(outputs, dim=1) == targets).sum().item()
                total_pixels += targets.numel()

        ious = np.nanmean(np.array(all_ious), axis=0)
        dices = np.nanmean(np.array(all_dices), axis=0)

        result = {
            "pixel_accuracy": total_correct / total_pixels,
            "mean_IoU": float(np.nanmean(ious[1:])) if self.n_classes > 1 else float(ious[0]),
            "per_class": {},
        }

        if class_names is None:
            class_names = [f"class_{i}" for i in range(self.n_classes)]

        for i, name in enumerate(class_names):
            result["per_class"][name] = {
                "IoU": float(ious[i]) if not np.isnan(ious[i]) else 0.0,
                "Dice": float(dices[i]) if not np.isnan(dices[i]) else 0.0,
            }

        return result

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """单张图像推理 → (segmentation_map, confidence_map)."""
        self.model.eval()

        # 预处理
        if image.ndim == 2:
            img = np.expand_dims(image, axis=0)
        elif image.ndim == 3 and image.shape[2] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if HAS_CV2 else image.mean(axis=2)
            img = np.expand_dims(img, axis=0)
        else:
            img = image

        img_t = torch.tensor(img.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_t)
            probs = F.softmax(logits, dim=1)
            seg_map = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            confidence = probs.max(dim=1)[0].squeeze(0).cpu().numpy()

        return seg_map, confidence

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "n_classes": self.n_classes,
            "n_channels": self.n_channels,
            "history": self.history,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.history = checkpoint.get("history", {})


# need for predict method
try:
    F = torch.nn.functional
except Exception:
    pass
