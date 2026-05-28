"""U-Net语义分割模型 — Dice+CE Loss + IoU/Dice Metrics"""

import numpy as np
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DoubleConv(nn.Module):
    """(Conv3x3 → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """下采样: MaxPool → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """上采样: Upsample/ConvTranspose → Concat(skip) → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸差异
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积: 1×1映射到类别数"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net语义分割网络.

    Args:
        n_channels: 输入通道数 (1=灰度, 3=RGB)
        n_classes: 输出类别数 (含背景)
        bilinear: 是否使用双线性上采样 (True=更快, False=更精确)
        base_channels: 基础通道数
    """

    def __init__(self, n_channels: int = 1, n_classes: int = 4,
                 bilinear: bool = True, base_channels: int = 64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        ch = base_channels
        self.inc = DoubleConv(n_channels, ch)
        self.down1 = Down(ch, ch * 2)       # 64→128
        self.down2 = Down(ch * 2, ch * 4)   # 128→256
        self.down3 = Down(ch * 4, ch * 8)   # 256→512
        factor = 2 if bilinear else 1
        self.down4 = Down(ch * 8, ch * 16 // factor)  # 512→1024 (bilinear) / 512→512
        self.up1 = Up(ch * 16, ch * 8 // factor, bilinear)
        self.up2 = Up(ch * 8, ch * 4 // factor, bilinear)
        self.up3 = Up(ch * 4, ch * 2 // factor, bilinear)
        self.up4 = Up(ch * 2, ch, bilinear)
        self.outc = OutConv(ch, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ================================================================
# 损失函数
# ================================================================

class DiceLoss(nn.Module):
    """Dice Loss for semantic segmentation."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        # target: (B, H, W) → one-hot (B, C, H, W)
        target_onehot = F.one_hot(target.long(), num_classes=pred.size(1))
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Dice Loss + CrossEntropy Loss 组合.

    总损失 = α * Dice + (1-α) * CE
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1e-5,
                 class_weights: List[float] = None):
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss(smooth)

        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
        else:
            weight = None
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        # CE需要: pred (B, C, H, W), target (B, H, W) as Long
        if target.dtype != torch.long:
            target = target.long()
        ce_loss = self.ce(pred, target)
        return self.alpha * dice_loss + (1.0 - self.alpha) * ce_loss


# ================================================================
# 评估指标
# ================================================================

def compute_iou(pred: torch.Tensor, target: torch.Tensor,
                n_classes: int, ignore_index: int = -100) -> np.ndarray:
    """计算每类IoU (Jaccard Index).

    Returns:
        (n_classes,) float array
    """
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    target = target.cpu().numpy()

    ious = []
    for cls in range(n_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.array(ious)


def compute_dice(pred: torch.Tensor, target: torch.Tensor,
                 n_classes: int, smooth: float = 1e-5) -> np.ndarray:
    """计算每类Dice系数."""
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    target = target.cpu().numpy()

    dices = []
    for cls in range(n_classes):
        pred_mask = (pred == cls).astype(np.float32)
        target_mask = (target == cls).astype(np.float32)
        intersection = (pred_mask * target_mask).sum()
        denom = pred_mask.sum() + target_mask.sum()
        if denom == 0:
            dices.append(float('nan'))
        else:
            dices.append(2.0 * intersection / (denom + smooth))
    return np.array(dices)


def mean_iou(pred: torch.Tensor, target: torch.Tensor,
             n_classes: int, ignore_bg: bool = True) -> float:
    """计算平均IoU (mIoU)."""
    ious = compute_iou(pred, target, n_classes)
    start = 1 if ignore_bg else 0
    valid = ious[start:][~np.isnan(ious[start:])]
    return float(valid.mean()) if len(valid) > 0 else 0.0


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """逐像素准确率."""
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total
