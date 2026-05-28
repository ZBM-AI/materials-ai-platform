"""模型诊断 v5 — 学习曲线、预测vs实际图、残差分布、适用域分析"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def generate_learning_curve_data(
    model, X: np.ndarray, y: np.ndarray,
    train_sizes: np.ndarray = None,
    cv: int = 5, n_jobs: int = -1
) -> dict:
    """生成学习曲线数据.

    返回 {train_sizes, train_scores_mean, train_scores_std,
           test_scores_mean, test_scores_std} 用于绘图.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv,
        scoring="neg_mean_absolute_error", n_jobs=n_jobs,
        shuffle=True, random_state=42,
    )
    return {
        "train_sizes": sizes_abs.tolist(),
        "train_mae_mean": (-train_scores.mean(axis=1)).tolist(),
        "train_mae_std": train_scores.std(axis=1).tolist(),
        "test_mae_mean": (-test_scores.mean(axis=1)).tolist(),
        "test_mae_std": test_scores.std(axis=1).tolist(),
    }


def generate_parity_data(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """生成预测vs实际散点数据 + 残差统计.

    返回可用于Streamlit plotly_chart或matplotlib的数据.
    """
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)
    return {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "residuals": residuals.tolist(),
        "abs_residuals": abs_residuals.tolist(),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "max_overestimate": float(np.max(residuals)),
        "max_underestimate": float(np.min(residuals)),
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "residual_skew": float(pd.Series(residuals).skew()),
    }


def diagnostic_report(model, X: np.ndarray, y: np.ndarray,
                      feature_names: Optional[List[str]] = None,
                      formulas: Optional[List[str]] = None) -> dict:
    """生成完整诊断报告: 学习曲线 + 残差分析 + 样本级诊断.

    返回包含所有诊断数据的字典.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model_copy = None
    try:
        from sklearn.base import clone
        model_copy = clone(model)
    except Exception:
        model_copy = model

    if model_copy is not None and hasattr(model_copy, 'fit'):
        model_copy.fit(X_train_s, y_train)
        y_pred = model_copy.predict(X_test_s)
    else:
        y_pred = model.predict(X_test_s) if hasattr(model, 'predict') else np.zeros_like(y_test)

    parity = generate_parity_data(y_test, y_pred)

    learning = None
    if model_copy is not None and hasattr(model_copy, 'fit'):
        try:
            learning = generate_learning_curve_data(model, X_train_s, y_train, cv=min(5, len(y_train)//10))
        except Exception:
            learning = None

    worst_indices = np.argsort(parity["abs_residuals"])[-10:][::-1]
    worst_samples = []
    if formulas:
        for idx in worst_indices:
            worst_samples.append({
                "formula": formulas[idx],
                "y_true": parity["y_true"][idx],
                "y_pred": parity["y_pred"][idx],
                "residual": parity["residuals"][idx],
            })

    report = {
        "parity": parity,
        "learning_curve": learning,
        "worst_samples": worst_samples,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X.shape[1],
    }

    return report


def applicability_domain(X_train: np.ndarray, X_new: np.ndarray,
                         threshold_percentile: float = 95.0) -> np.ndarray:
    """适用域分析: 基于马氏距离判断新样本是否在训练分布内.

    Args:
        X_train: 训练集特征矩阵
        X_new: 新样本特征矩阵
        threshold_percentile: 距离阈值百分位

    Returns:
        inside_domain: bool数组, True表示在适用域内
        distances: 马氏距离数组
    """
    from scipy.spatial.distance import cdist
    mean_vec = X_train.mean(axis=0).reshape(1, -1)
    cov = np.cov(X_train.T)
    try:
        cov_inv = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    diff = X_new - mean_vec
    distances = np.sqrt(np.sum((diff @ cov_inv) * diff, axis=1))
    threshold = np.percentile(
        np.sqrt(np.sum(((X_train - mean_vec) @ cov_inv) * (X_train - mean_vec), axis=1)),
        threshold_percentile
    )
    inside_domain = distances <= threshold
    return inside_domain, distances


def plot_learning_curve(learning_data: dict, ax=None):
    """绘制学习曲线 (matplotlib)."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    sizes = learning_data["train_sizes"]
    ax.plot(sizes, learning_data["train_mae_mean"], "b-", label="Training MAE")
    ax.fill_between(
        sizes,
        np.array(learning_data["train_mae_mean"]) - np.array(learning_data["train_mae_std"]),
        np.array(learning_data["train_mae_mean"]) + np.array(learning_data["train_mae_std"]),
        alpha=0.15, color="b"
    )
    ax.plot(sizes, learning_data["test_mae_mean"], "r-", label="Validation MAE")
    ax.fill_between(
        sizes,
        np.array(learning_data["test_mae_mean"]) - np.array(learning_data["test_mae_std"]),
        np.array(learning_data["test_mae_mean"]) + np.array(learning_data["test_mae_std"]),
        alpha=0.15, color="r"
    )
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("MAE")
    ax.set_title("Learning Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_parity(parity_data: dict, ax=None):
    """绘制预测vs实际散点图 (matplotlib)."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    y_true = parity_data["y_true"]
    y_pred = parity_data["y_pred"]
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="none")
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Ideal")
    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title(f"Parity Plot (R²={parity_data['r2']:.3f}, MAE={parity_data['mae']:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_residuals(parity_data: dict, ax=None):
    """绘制残差分布直方图 (matplotlib)."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    residuals = parity_data["residuals"]
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.axvline(np.mean(residuals), color="orange", linestyle="-", linewidth=1,
               label=f"Mean={parity_data['mean_residual']:.4f}")
    ax.set_xlabel("Residual (True - Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"Residual Distribution "
        f"(Std={parity_data['std_residual']:.4f}, Skew={parity_data['residual_skew']:.3f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def save_diagnostic_plots(report: dict, output_dir: str):
    """将诊断图保存为PNG文件."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    if report.get("learning_curve"):
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_learning_curve(report["learning_curve"], ax)
        path = os.path.join(output_dir, "learning_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["learning_curve"] = path

    if report.get("parity"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_parity(report["parity"], ax1)
        plot_residuals(report["parity"], ax2)
        path = os.path.join(output_dir, "parity_residuals.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["parity_residuals"] = path

    return paths
