"""模块3: 性能预测 v5 — GPR + MEGNet + 热导率/屈服强度 + 全特征 + 诊断图"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.property_prediction.models import PropertyPredictor
import config

st.set_page_config(page_title="性能预测 v5", page_icon="🔮", layout="wide")
st.title("🔮 材料性能预测 v5 — GPR + MEGNet + 热导率/屈服强度")
st.markdown("RF · XGBoost · GPR · CGCNN · MEGNet | SHAP | 主动学习 | FastAPI | MatBench | One-Hot")

st.markdown("---")


@st.cache_resource
def load_models():
    models = {}
    shear_path = config.MECHANICAL_MODEL.replace("mechanical", "shear_modulus")
    for key, path in [
        ("band_gap", config.BAND_GAP_MODEL),
        ("formation_energy", config.FORMATION_ENERGY_MODEL),
        ("bulk_modulus", config.MECHANICAL_MODEL),
        ("shear_modulus", shear_path),
        ("thermal_conductivity", config.THERMAL_CONDUCTIVITY_MODEL),
        ("yield_strength", config.YIELD_STRENGTH_MODEL),
    ]:
        if os.path.exists(path):
            try:
                models[key] = PropertyPredictor.load(path)
            except Exception:
                pass
    return models


models = load_models()

tabs = st.tabs([
    "🎯 预测", "📊 模型对比", "📈 诊断图",
    "🏆 MatBench", "🔍 SHAP", "🧬 特征工程", "🧪 主动学习", "🚀 API",
])

# ═══════════════════════════════════════════════════════════
# Tab 1: Prediction
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("单材料预测")
        formula = st.text_input("化学式", placeholder="Fe2O3, SrTiO3, CsPbI3...",
                                label_visibility="collapsed")
        predict_btn = st.button("预测性能", type="primary", use_container_width=True)
        st.write("**示例:** " + " | ".join(
            f"`{f}`" for f in ["TiO2", "Fe2O3", "SrTiO3", "ZnO", "GaN",
                               "CsPbI3", "Al2O3", "MoS2", "SiC", "Y3Al5O12"]
        ))
        if predict_btn and formula:
            st.markdown("---")
            for key, model in models.items():
                try:
                    pred, std = model.predict_with_std(formula)
                    labels = {
                        "band_gap": ("带隙", "eV"),
                        "formation_energy": ("形成能", "eV/atom"),
                        "bulk_modulus": ("体模量", "GPa"),
                        "shear_modulus": ("剪切模量", "GPa"),
                        "thermal_conductivity": ("热导率", "W/(m*K)"),
                        "yield_strength": ("屈服强度", "MPa"),
                    }
                    label, unit = labels.get(key, (key, ""))
                    decimals = 2 if key == "yield_strength" else 4
                    st.metric(f"{label}", f"{pred:.{decimals}f} {unit}",
                              delta=f"+/- {std:.{decimals}f} {unit}" if std > 0 else None)
                except Exception:
                    st.error(f"{key}: 预测失败")
        elif predict_btn and not formula:
            st.error("请输入化学式!")

    with col_right:
        st.subheader("批量预测")
        batch_input = st.text_area(
            "化学式列表", placeholder="TiO2\nFe2O3\nZnO\nSrTiO3\nGaN", height=200,
        )
        lines = [l.strip() for l in batch_input.replace(',', '\n').split('\n') if l.strip()]
        if lines and st.button("批量预测", type="primary"):
            df = pd.DataFrame({"化学式": lines})
            for key, model in models.items():
                try:
                    preds = model.predict_batch(lines)
                    labels = {
                        "band_gap": "带隙_eV", "formation_energy": "形成能_eV_per_atom",
                        "bulk_modulus": "体模量_GPa", "shear_modulus": "剪切模量_GPa",
                        "thermal_conductivity": "热导率_W_per_mK",
                        "yield_strength": "屈服强度_MPa",
                    }
                    df[labels.get(key, key)] = preds
                except Exception:
                    pass
            st.dataframe(df, use_container_width=True)
            st.download_button("下载 CSV", df.to_csv(index=False),
                               file_name="predictions.csv", mime="text/csv")

    st.markdown("---")
    st.caption("v5: GPR+MEGNet + 热导率/屈服强度 + One-Hot编码 + MatBench + 诊断图")

# ═══════════════════════════════════════════════════════════
# Tab 2: Model Comparison (RF vs XGBoost vs GPR vs CGCNN)
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("多模型对比 — RF vs XGBoost vs GPR vs CGCNN vs MEGNet")

    col_info, col_btn = st.columns([2, 1])
    with col_info:
        st.markdown("""
        | 模型 | 优势 | 不确定性 |
        |------|------|---------|
        | **Random Forest** | 集成学习, 鲁棒 | 树级标准差 |
        | **XGBoost** | 梯度提升, 精度最高 | 无 (需外部分位数回归) |
        | **Gaussian Process** | 贝叶斯推断, 小样本 | **原生后验方差** |
        | **CGCNN** | 图卷积, 结构感知 | 无 |
        | **MEGNet** | 全局状态+边更新, 通用GNN | 无 |
        """)
    with col_btn:
        n_samples_slider = st.slider("样本数", 100, 318, 150, 50)
        include_gpr = st.checkbox("包含 Gaussian Process", value=True)
        include_megnet = st.checkbox("包含 MEGNet (慢)", value=False)
        run_comp = st.button("运行模型对比 (v5)", type="primary")

    if run_comp:
        with st.spinner("训练中... RF → XGBoost → GPR → CGCNN → MEGNet..."):
            try:
                from modules.property_prediction.model_comparison import run_model_comparison
                df_data = pd.read_csv(config.BAND_GAP_DATA)
                formulas = df_data["formula"].tolist()[:n_samples_slider]
                targets = df_data["band_gap_eV"].values[:n_samples_slider]
                comp_df, _ = run_model_comparison(formulas, targets, "band_gap_eV",
                                                  include_gpr=include_gpr,
                                                  include_megnet=include_megnet)

                display_cols = ["model", "test_r2", "test_mae", "test_rmse",
                                "cv_r2_mean", "cv_mae_mean", "time_seconds"]
                if include_gpr:
                    display_cols += ["mean_uncertainty", "calibration_ratio"]
                display_cols = [c for c in display_cols if c in comp_df.columns]

                display_df = comp_df[display_cols].copy()
                for col in display_df.select_dtypes(include=['float64', 'float32']).columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                    )
                st.dataframe(display_df, use_container_width=True)

                valid = comp_df[comp_df["test_r2"].notna()]
                if len(valid) > 0:
                    best = valid.iloc[valid["test_r2"].argmax()]
                    st.success(
                        f"最佳: **{best['model']}** "
                        f"(Test R²={best['test_r2']:.4f}, MAE={best['test_mae']:.4f})"
                    )

                if include_gpr and "calibration_ratio" in comp_df.columns:
                    gpr_rows = comp_df[comp_df["model"] == "GaussianProcess"]
                    if len(gpr_rows) > 0:
                        cal = gpr_rows.iloc[0]["calibration_ratio"]
                        st.info(
                            f"**GPR不确定性校准**: {cal:.1%} 的真实值落在预测值 ±1σ 范围内 "
                            f"(理想值: 68%)"
                        )

                st.caption("""
                **GPR注意事项**: 高斯过程在N>1000时计算复杂度O(N³)极高。
                对于大规模数据集, 建议使用稀疏GPR (SparseGP) 或仅用小样本子集训练。
                """)
            except Exception as e:
                st.error(f"对比失败: {e}")
                import traceback
                st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════
# Tab 3: Diagnostic Plots
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("模型诊断 — 学习曲线 + 预测vs实际 + 残差分布")

    col_diag1, col_diag2 = st.columns([1, 2])
    with col_diag1:
        st.write("**诊断说明**")
        st.markdown("""
        - **学习曲线**: 训练/验证误差vs训练样本数。
          高训练误差+高验证误差 = 欠拟合;
          低训练误差+高验证误差 = 过拟合。
        - **预测vs实际图**: 散点越接近对角线越好。
          系统性偏离表示模型有偏差。
        - **残差分布**: 应以0为中心的正态分布。
          偏斜表示模型系统性高估或低估。
        """)
        diag_btn = st.button("生成诊断图", type="primary")

    with col_diag2:
        if diag_btn:
            with st.spinner("生成诊断图中..."):
                try:
                    from modules.property_prediction.diagnostics import (
                        diagnostic_report, save_diagnostic_plots,
                    )
                    from modules.property_prediction.features_v4 import MagpieFeaturizer
                    from sklearn.ensemble import RandomForestRegressor

                    df_data = pd.read_csv(config.BAND_GAP_DATA)
                    formulas = df_data["formula"].tolist()[:150]
                    targets = df_data["band_gap_eV"].values[:150]

                    featurizer = MagpieFeaturizer()
                    X = featurizer.featurize_batch(formulas)
                    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)

                    report = diagnostic_report(model, X, targets, formulas=formulas)

                    diag_dir = os.path.join(config.DATA_DIR, "diagnostics")
                    paths = save_diagnostic_plots(report, diag_dir)

                    if paths.get("learning_curve"):
                        st.image(paths["learning_curve"], caption="学习曲线",
                                 use_container_width=True)
                    if paths.get("parity_residuals"):
                        st.image(paths["parity_residuals"],
                                 caption="预测vs实际 + 残差分布",
                                 use_container_width=True)

                    if report.get("worst_samples"):
                        st.write("**预测最差的10个样本**")
                        worst_df = pd.DataFrame(report["worst_samples"])
                        st.dataframe(worst_df, use_container_width=True)

                    st.success(
                        f"R²={report['parity']['r2']:.3f}, "
                        f"MAE={report['parity']['mae']:.3f}, "
                        f"残差均值={report['parity']['mean_residual']:.4f}, "
                        f"残差偏度={report['parity']['residual_skew']:.3f} "
                        f"({'无系统性偏差' if abs(report['parity']['residual_skew']) < 0.5 else '存在系统性偏差'})"
                    )
                except Exception as e:
                    st.error(f"诊断失败: {e}")

# ═══════════════════════════════════════════════════════════
# Tab 4: MatBench Benchmark
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("MatBench 基准测试")
    st.markdown("""
    [MatBench](https://matbench.materialsproject.org/) 是材料科学ML的标准化评测基准。
    将下载的JSON文件放入 `data/matbench_cache/` 即可离线使用。
    """)

    from modules.property_prediction.matbench_benchmark import (
        SOTA_COMPARISON, AVAILABLE_TASKS, MATBENCH_AVAILABLE as MB_AVAIL,
        list_available_local_datasets,
    )

    local_datasets = list_available_local_datasets()

    st.write("**MatBench任务状态**")
    sota_rows = []
    for task_id, task_name in AVAILABLE_TASKS.items():
        info = SOTA_COMPARISON.get(task_name, {})
        available = task_name in local_datasets
        sota_rows.append({
            "任务ID": task_id,
            "描述": info.get("description", task_name),
            "总样本数": info.get("n_samples", "?"),
            "本地可用": "Yes" if available else "需下载",
            "SOTA模型": info.get("sota_model", "?"),
            "SOTA MAE": info.get("sota_mae", "?"),
        })
    st.dataframe(pd.DataFrame(sota_rows), use_container_width=True)

    if local_datasets:
        st.success(f"**已下载**: {', '.join([d for d in local_datasets])} ({len(local_datasets)}/{len(AVAILABLE_TASKS)} 任务)")
    else:
        st.warning("未检测到本地数据集。将JSON文件下载到 `data/matbench_cache/` 目录。")
        st.caption("""
        下载方法: 访问 https://ml.materialsproject.org/projects/ 搜索 "matbench_xxx.json.gz",
        或将已下载的JSON文件放入 `materials_ai/data/matbench_cache/` 目录。
        支持 .json 和 .json.gz 格式。
        """)

    st.markdown("---")
    st.write("**运行基准测试**")
    col_mb1, col_mb2 = st.columns([1, 2])
    with col_mb1:
        # Only show available tasks
        available_task_ids = [
            tid for tid, tname in AVAILABLE_TASKS.items()
            if tname in local_datasets
        ]
        if not available_task_ids:
            available_task_ids = list(AVAILABLE_TASKS.keys())

        task_select = st.selectbox(
            "选择任务", available_task_ids,
            format_func=lambda x: f"{x} {'(本地可用)' if AVAILABLE_TASKS[x] in local_datasets else '(需下载)'}"
        )
        n_mb = st.slider("样本数 (MatBench)", 200, 20000, 1000, 200,
                         help="样本越多越接近SOTA, 但训练更慢")
        mb_model = st.radio("模型", ["random_forest", "xgboost"],
                            format_func=lambda x: "Random Forest" if x == "random_forest" else "XGBoost")
        mb_btn = st.button("运行MatBench评测", type="primary")

    with col_mb2:
        if mb_btn:
            task_name = AVAILABLE_TASKS[task_select]
            available = task_name in local_datasets
            if not available:
                st.warning(f"{task_name} 未下载。请先下载数据集。")

            with st.spinner(f"加载MatBench数据 + 训练 {mb_model} (最多{n_mb}样本)..."):
                try:
                    from modules.property_prediction.matbench_benchmark import benchmark_on_matbench

                    def make_model():
                        if mb_model == "xgboost":
                            import xgboost as xgb
                            return xgb.XGBRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
                        from sklearn.ensemble import RandomForestRegressor
                        return RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)

                    result = benchmark_on_matbench(task_name, make_model, n_samples=n_mb)

                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Test MAE", f"{result['test_mae']:.4f}",
                                  delta=f"SOTA: {result.get('sota_mae','?')}")
                    with col_m2:
                        st.metric("Test R²", f"{result['test_r2']:.4f}")
                    with col_m3:
                        st.metric("Test RMSE", f"{result['test_rmse']:.4f}")

                    if result.get("sota_mae"):
                        gap = result["mae_gap_to_sota"]
                        st.metric(
                            f"MAE差距 vs SOTA ({result.get('sota_model','?')})",
                            f"{result['test_mae']:.4f}",
                            delta=f"{gap:+.4f} (Baseline: {result.get('baseline_mae','?')})",
                        )
                    st.caption(
                        f"训练样本: {result['n_train']} | 测试样本: {result['n_test']} | "
                        f"训练时间: {result['training_time_s']:.1f}s | "
                        f"特征维度: {result['feature_dim']}"
                    )
                except Exception as e:
                    st.error(f"评测失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ═══════════════════════════════════════════════════════════
# Tab 5: SHAP (simplified)
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("SHAP 模型可解释性")

    shap_dir = os.path.join(config.DATA_DIR, "shap_output")
    shap_summary = os.path.join(shap_dir, "shap_summary.png")
    shap_importance = os.path.join(shap_dir, "shap_importance.png")

    if os.path.exists(shap_summary):
        col1, col2 = st.columns(2)
        with col1:
            st.image(shap_summary, caption="SHAP Summary Plot", use_container_width=True)
        with col2:
            st.image(shap_importance, caption="SHAP Feature Importance", use_container_width=True)

        st.success("**electronegativity_std** 是带隙预测最主导的特征 (SHAP值比其他高 ~10倍)")
    else:
        st.info("SHAP图尚未生成。运行模型对比后自动生成。")

# ═══════════════════════════════════════════════════════════
# Tab 6: Feature Engineering
# ═══════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("特征工程方法对比")
    st.markdown("""
    **四种特征表示方法**, 从不同角度编码材料化学成分:
    """)

    feat_tab1, feat_tab2, feat_tab3 = st.tabs([
        "One-Hot 编码", "元素属性矩阵", "Magpie 统计",
    ])

    with feat_tab1:
        st.write("**One-Hot 成分编码 (95维)**")
        st.markdown("""
        将化学式编码为固定长度向量: 每个元素(Z=1~95)占一个位置,
        值为该元素在化学式中的**原子分数**。

        - 维度: 95 (H ~ Am)
        - 稀疏表示: 简单化合物仅数个非零值
        - 适合: 线性模型, 神经网络第一层
        """)
        example = st.text_input("化学式示例", value="SrTiO3", key="onehot_example")
        if example:
            try:
                from modules.property_prediction.features_v5 import OneHotCompositionFeaturizer
                oh = OneHotCompositionFeaturizer()
                vec = oh.featurize(example)
                nonzero = [(oh.get_feature_names()[i], f"{vec[i]:.3f}")
                          for i in range(len(vec)) if vec[i] > 0.001]
                st.write(f"非零元素: {len(nonzero)}")
                st.dataframe(pd.DataFrame(nonzero, columns=["元素", "分数"]), use_container_width=True)

                st.write("**向量可视化**")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.bar(range(len(vec)), vec, width=1.0)
                ax.set_xlabel("Atomic Number")
                ax.set_ylabel("Fraction")
                ax.set_title(f"One-Hot Encoding: {example}")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"编码失败: {e}")

    with feat_tab2:
        st.write("**元素属性矩阵**")
        st.markdown("""
        将化学式表示为 **(max_elements, n_properties)** 的2D矩阵,
        每行对应一个元素, 列为其基本属性。

        - 形状: (8, 11) — 最多8种元素 × 11个属性
        - 属性: 原子序数、电负性、原子半径、电离能、电子亲和能等
        - 适合: CNN、Attention、GNN等需要空间结构的模型
        """)
        example2 = st.text_input("化学式示例", value="CsPbI3", key="matrix_example")
        if example2:
            try:
                from modules.property_prediction.features_v5 import ElementPropertyMatrix
                epm = ElementPropertyMatrix(max_elements=5)
                mat = epm.featurize(example2)
                prop_names = epm.get_property_names()
                st.dataframe(
                    pd.DataFrame(mat, columns=prop_names)
                    .style.format("{:.3f}")
                    .background_gradient(cmap="Blues", axis=1),
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"编码失败: {e}")

    with feat_tab3:
        st.write("**Magpie 特征 (87维)**")
        st.markdown("""
        对每种元素属性统计: mean、max、min、std、weighted_avg (5×10=50维)
        + 化学计量特征 (6维) + 块级特征 (5维) + 化合物描述符 (10维) + 工艺参数 (16维)

        - 适合: 树模型 (RF、XGBoost)
        - 特点: 捕获元素属性分布, 但不保留空间信息
        """)

    with st.expander("特征方法对比总结"):
        st.markdown("""
        | 方法 | 维度 | 空间信息 | 适合模型 | 缺点 |
        |------|------|---------|---------|------|
        | **One-Hot** | 95 | 无 | 线性、MLP | 稀疏、不捕获跨元素关系 |
        | **元素属性矩阵** | (N, 11) | 有(元素序) | CNN、Attention | 需固定矩阵大小 |
        | **Magpie** | 87 | 无 | RF、XGBoost | 丢失元素个体信息 |
        | **Combined** | 192 | 混合 | 所有模型 | 维度较高 |
        """)

# ═══════════════════════════════════════════════════════════
# Tab 7: Active Learning
# ═══════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("主动学习 — 不确定性采样")
    st.markdown("""
    **流程**: 训练模型 → 预测候选池 → 选不确定性最高的 → 实验合成 → 加入训练集 → 重复

    **GPR优势**: Gaussian Process 提供原生贝叶斯后验方差,
    比RF树级标准差更可靠的主动学习不确定性估计。
    """)

    with st.expander("主动学习伪代码"):
        st.code("""
L = [(x_i, y_i)]    # 已标注数据
U = [x_j]            # 候选池
B = 50               # 实验预算

for iter in range(B):
    model.fit(L.X, L.y)
    mu, sigma = model.predict(U.X, return_std=True)  # GPR原生支持
    next_idx = argmax(sigma)          # 不确定性采样
    x_next, y_next = U[next_idx], run_experiment(U[next_idx])
    L.append((x_next, y_next))
    U.remove(x_next)
        """, language="python")

# ═══════════════════════════════════════════════════════════
# Tab 8: API
# ═══════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("FastAPI 模型服务")
    st.markdown("启动: `uvicorn modules.property_prediction.api:app --host 0.0.0.0 --port 8000`")

    col_api1, col_api2 = st.columns(2)
    with col_api1:
        st.write("**请求**")
        st.code("""POST /predict
{
  "compositions": ["TiO2", "Fe2O3"],
  "target": "band_gap",
  "process": {"temperature": 1073, "method": "solid_state"},
  "return_std": true
}""", language="json")
    with col_api2:
        st.write("**响应**")
        st.code("""{
  "target": "band_gap", "unit": "eV",
  "predictions": [{
    "composition": "TiO2",
    "predicted_value": 3.1245,
    "std_deviation": 0.2341,
    "confidence_68": [2.8904, 3.3586],
    "confidence_95": [2.6563, 3.5927]
  }]
}""", language="json")
