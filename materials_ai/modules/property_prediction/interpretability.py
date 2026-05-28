"""模型可解释性 — 基于SHAP的XGBoost特征重要性 & 依赖图"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass


def shap_analysis(model, X: np.ndarray, feature_names: List[str],
                  X_background: Optional[np.ndarray] = None,
                  max_display: int = 20,
                  output_dir: str = "") -> dict:
    """对XGBoost/RF模型进行SHAP分析, 生成特征重要性图和依赖图.

    Args:
        model: 训练好的XGBoost或RF模型
        X: 特征矩阵 (用于计算SHAP值)
        feature_names: 特征名列表
        X_background: 背景数据 (默认取X的前100行)
        max_display: 显示前N个特征
        output_dir: 图片输出目录

    Returns:
        dict: {"importance": [...], "shap_values": np.ndarray, "fig_paths": {...}}
    """
    if not SHAP_AVAILABLE:
        return {"error": "shap not installed. Run: pip install shap"}

    if X_background is None:
        X_background = X[:min(100, len(X))]

    result = {"importance": [], "shap_values": None, "fig_paths": {}, "plots_html": ""}

    try:
        explainer = shap.TreeExplainer(model, X_background)
        shap_values = explainer.shap_values(X)
        result["shap_values"] = shap_values

        if shap_values.ndim == 3:
            shap_values_for_importance = shap_values[:, :, 0]
        else:
            shap_values_for_importance = shap_values

        mean_abs_shap = np.abs(shap_values_for_importance).mean(axis=0)
        importance_pairs = list(zip(feature_names, mean_abs_shap.tolist()))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        result["importance"] = importance_pairs

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            plt_available = False
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt_available = True
            except ImportError:
                pass

            if plt_available:
                fig_path = os.path.join(output_dir, "shap_summary.png")
                shap.summary_plot(
                    shap_values_for_importance, X, feature_names=feature_names,
                    max_display=max_display, show=False
                )
                plt.tight_layout()
                plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close()
                result["fig_paths"]["summary"] = fig_path

                fig_path2 = os.path.join(output_dir, "shap_importance.png")
                shap.summary_plot(
                    shap_values_for_importance, X, feature_names=feature_names,
                    plot_type="bar", max_display=max_display, show=False
                )
                plt.tight_layout()
                plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
                plt.close()
                result["fig_paths"]["importance_bar"] = fig_path2

                top_features = [feature_names[i] for i in
                                np.argsort(mean_abs_shap)[-4:][::-1]]
                for feat in top_features:
                    idx = feature_names.index(feat)
                    fig_path = os.path.join(
                        output_dir, f"shap_dependence_{feat.replace('/', '_')}.png"
                    )
                    shap.dependence_plot(
                        idx, shap_values_for_importance, X,
                        feature_names=feature_names, show=False
                    )
                    plt.tight_layout()
                    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                    plt.close()
                    result["fig_paths"][f"dependence_{feat}"] = fig_path

    except Exception as e:
        result["error"] = str(e)

    return result


def generate_shap_report(model, X: np.ndarray, feature_names: List[str],
                         output_dir: str) -> str:
    """生成SHAP分析文本报告"""
    result = shap_analysis(model, X, feature_names, output_dir=output_dir)
    if "error" in result:
        return f"[!] SHAP analysis error: {result['error']}"

    lines = [
        "=" * 60,
        "SHAP Feature Importance Report (XGBoost)",
        "=" * 60,
        "",
        "Top 15 Features (mean |SHAP|):",
        "-" * 40,
    ]
    for i, (feat, imp) in enumerate(result["importance"][:15], 1):
        lines.append(f"  {i:2d}. {feat:<35s} {imp:.6f}")
    lines.append("")
    if result["fig_paths"]:
        lines.append("Generated figures:")
        for key, path in result["fig_paths"].items():
            lines.append(f"  - {key}: {path}")
    return "\n".join(lines)
