"""模块4: 微观图像智能分析 — Streamlit 页面"""

import streamlit as st
import os
import sys
import numpy as np
import tempfile
from io import BytesIO
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from modules.microscopy_analysis.preprocessing import MicrographPreprocessor
from modules.microscopy_analysis.phase_segmenter import PhaseSegmenter
from modules.microscopy_analysis.grain_detector import GrainDetector
from modules.microscopy_analysis.defect_classifier import DefectAnalyzer
from modules.microscopy_analysis.structure_classifier import MicrostructureClassifier
from modules.microscopy_analysis.report_generator import MicrographReport
import config

st.set_page_config(page_title="显微图像分析", page_icon="🔬", layout="wide")

st.title("🔬 微观图像智能分析")
st.markdown("U-Net物相分割 · YOLOv8晶粒检测 · 缺陷分类 · 组织识别 · 定量统计报告")

if not HAS_CV2:
    st.error("OpenCV 未安装。请运行: `pip install opencv-python`")
    st.stop()

# ---- 侧边栏: 参数控制 ----
with st.sidebar:
    st.header("⚙️ 分析参数")

    st.subheader("标尺校准")
    pixel_scale = st.number_input(
        "像素比例 (μm/px)", value=config.DEFAULT_PIXEL_SCALE_UM,
        min_value=0.001, max_value=10.0, step=0.01, format="%.3f",
        help="显微镜标尺: μm per pixel"
    )

    st.subheader("预处理")
    apply_denoise = st.checkbox("去噪", value=True)
    denoise_strength = st.selectbox("去噪强度", ["low", "medium", "high"], index=0,
                                    help="low=高斯模糊, medium/high=NLM去噪")
    apply_equalize = st.checkbox("CLAHE 光照均衡化", value=True)

    st.subheader("检测阈值")
    min_grain_area = st.slider("最小晶粒面积 (px)", 50, 500, config.DEFAULT_MIN_GRAIN_AREA_PX, 50)
    min_defect_area = st.slider("最小缺陷面积 (px)", 5, 200, config.DEFAULT_MIN_DEFECT_AREA_PX, 10)

    st.markdown("---")
    st.caption(f"模型目录: `{config.MICROSCOPY_MODELS_DIR}`")

# ---- 模型加载 ----
@st.cache_resource
def load_models():
    phase_segmenter = PhaseSegmenter(
        model_path=config.UNET_MODEL_PATH if os.path.exists(config.UNET_MODEL_PATH) else None,
        n_classes=config.UNET_N_CLASSES,
    )
    grain_detector = GrainDetector(
        model_path=config.YOLO_GRAIN_MODEL_PATH if os.path.exists(config.YOLO_GRAIN_MODEL_PATH) else None,
    )
    defect_analyzer = DefectAnalyzer(
        cnn_model_path=config.DEFECT_CNN_MODEL_PATH if os.path.exists(config.DEFECT_CNN_MODEL_PATH) else None,
    )
    structure_classifier = MicrostructureClassifier(
        model_path=config.STRUCTURE_CNN_MODEL_PATH if os.path.exists(config.STRUCTURE_CNN_MODEL_PATH) else None,
    )
    preprocessor = MicrographPreprocessor()
    return phase_segmenter, grain_detector, defect_analyzer, structure_classifier, preprocessor

phase_segmenter, grain_detector, defect_analyzer, structure_classifier, preprocessor = load_models()

# ---- 图像上传 ----
st.markdown("---")
upload_col, preview_col = st.columns([1, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "上传显微图像 (SEM/TEM/金相)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="支持常见显微镜图像格式"
    )

image = None
if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    with preview_col:
        st.image(image, caption=f"原始图像: {uploaded_file.name} ({image.shape[1]}×{image.shape[0]})",
                 use_container_width=True)

if image is None:
    st.info("👆 请上传一张显微图像开始分析")
    st.stop()

# ---- 运行分析 ----
run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    run_analysis = st.button("🚀 运行全部分析", type="primary", use_container_width=True)

if not run_analysis:
    st.info("点击 **运行全部分析** 开始处理图像")
    st.stop()

# 预处理
with st.spinner("预处理中..."):
    gray = preprocessor.process(
        image, denoise=apply_denoise, equalize=apply_equalize,
        denoise_strength=denoise_strength,
    )

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧪 物相分割", "🔲 晶粒分析", "⚠️ 缺陷检测", "🏗️ 组织分类", "📊 综合报告",
])

# ---- Tab 1: 物相分割 ----
with tab1:
    st.subheader("物相分割 (Phase Segmentation)")
    with st.spinner("物相分割中..."):
        phase_result = phase_segmenter.segment(image)

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(image, caption="原始图像", use_container_width=True)
    with col_b:
        seg_img = phase_result.get("segmented_image")
        if seg_img is not None:
            st.image(seg_img, caption=f"分割结果 ({phase_result.get('method', '')})",
                     use_container_width=True)

    fractions = phase_result.get("phase_fractions", {})
    if fractions:
        st.markdown("### 相比例")
        cols = st.columns(len(fractions))
        for i, (phase, frac) in enumerate(fractions.items()):
            pct = frac * 100
            with cols[i]:
                st.metric(label=phase, value=f"{pct:.2f}%")

        # 柱状图
        import plotly.graph_objects as go
        fig = go.Figure(data=[
            go.Bar(x=list(fractions.keys()), y=[v * 100 for v in fractions.values()],
                   marker_color=["#e74c3c", "#2ecc71", "#3498db", "#f39c12"][:len(fractions)],
                   text=[f"{v*100:.1f}%" for v in fractions.values()], textposition="auto")
        ])
        fig.update_layout(title="相面积比例 (%)", yaxis_title="Area %", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # 详细统计
    seg_map = phase_result.get("segmentation_map")
    if seg_map is not None:
        stats = phase_segmenter.compute_phase_statistics(seg_map, pixel_scale)
        if stats:
            st.markdown("### 各相统计")
            st.dataframe(stats, use_container_width=True)

# ---- Tab 2: 晶粒分析 ----
with tab2:
    st.subheader("晶粒检测与尺寸测量 (Grain Analysis)")
    with st.spinner("晶粒检测中..."):
        grain_result = grain_detector.detect_grains(image, pixel_scale_um=pixel_scale,
                                                     min_grain_area_px=min_grain_area)

    grain_img = grain_result.get("annotated_image")
    if grain_img is not None:
        st.image(grain_img, caption=f"晶粒边界 ({grain_result.get('method', '')})",
                 use_container_width=True)

    st.markdown("### 晶粒统计")
    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("检测晶粒数", grain_result.get("num_grains", 0))
    with metric_cols[1]:
        st.metric("平均晶粒尺寸 (μm)", f"{grain_result.get('avg_grain_size_um', 0):.2f}")
    with metric_cols[2]:
        st.metric("ASTM 晶粒度", f"G = {grain_result.get('grain_size_astm', 0)}")
    with metric_cols[3]:
        st.metric("截线法截距 (μm)", f"{grain_result.get('intercept_length_um', 0):.2f}")
    with metric_cols[4]:
        st.metric("检测方法", grain_result.get("method", "N/A").upper())

    areas_um2 = grain_result.get("grain_areas_um2", [])
    diams_um = grain_result.get("grain_diameters_um", [])
    if areas_um2 and diams_um:
        st.markdown("### 晶粒尺寸分布")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=["粒径分布直方图", "面积分布直方图"])

        fig.add_trace(
            go.Histogram(x=diams_um, nbinsx=30, marker_color="#3498db",
                         name="Diameter (μm)"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Histogram(x=areas_um2, nbinsx=30, marker_color="#2ecc71",
                         name="Area (μm²)"),
            row=1, col=2,
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # 汇总表
        st.markdown("### 尺寸统计")
        stats_df = {
            "统计量": ["Mean", "Std", "Min", "Max", "Median"],
            "面积 (μm²)": [
                f"{np.mean(areas_um2):.2f}", f"{np.std(areas_um2):.2f}",
                f"{np.min(areas_um2):.2f}", f"{np.max(areas_um2):.2f}",
                f"{np.median(areas_um2):.2f}",
            ],
            "直径 (μm)": [
                f"{np.mean(diams_um):.2f}", f"{np.std(diams_um):.2f}",
                f"{np.min(diams_um):.2f}", f"{np.max(diams_um):.2f}",
                f"{np.median(diams_um):.2f}",
            ],
        }
        st.dataframe(stats_df, use_container_width=True)

# ---- Tab 3: 缺陷检测 ----
with tab3:
    st.subheader("缺陷检测与分类 (Defect Analysis)")
    with st.spinner("缺陷检测中..."):
        defect_result = defect_analyzer.analyze(image, min_defect_area_px=min_defect_area)

    def_img = defect_result.get("annotated_image")
    if def_img is not None:
        st.image(def_img, caption="缺陷标注 (红:孔洞 | 蓝:裂纹 | 黄:夹杂物)",
                 use_container_width=True)

    st.markdown("### 缺陷统计")
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.metric("总缺陷数", defect_result.get("total_defects", 0))
    with dc2:
        st.metric("缺陷面积占比", f"{defect_result.get('defect_fraction', 0)*100:.3f}%")
    with dc3:
        total = defect_result.get("total_defects", 0)
        n_pores = len(defect_result.get("pores", []))
        n_cracks = len(defect_result.get("cracks", []))
        n_inclusions = len(defect_result.get("inclusions", []))
        st.metric("类型分布", f"孔洞:{n_pores} 裂纹:{n_cracks} 夹杂:{n_inclusions}")

    # 汇总表
    summary_rows = defect_analyzer.defect_summary_table(defect_result, pixel_scale)
    if summary_rows:
        st.markdown("### 缺陷类型汇总")
        st.dataframe(summary_rows, use_container_width=True)

    # 按类型展开详情
    for dtype, label, color in [("pores", "孔洞", "#e74c3c"),
                                  ("cracks", "裂纹", "#3498db"),
                                  ("inclusions", "夹杂物", "#f39c12")]:
        items = defect_result.get(dtype, [])
        if items:
            with st.expander(f"{label} ({len(items)} 个)", expanded=False):
                for j, item in enumerate(items[:50]):
                    st.markdown(
                        f"**#{j+1}** | 面积: {item['area_px']} px² | "
                        f"圆形度: {item['circularity']:.3f} | "
                        f"长宽比: {item['aspect_ratio']:.2f} | "
                        f"位置: ({item['centroid'][0]:.0f}, {item['centroid'][1]:.0f})"
                    )

# ---- Tab 4: 组织分类 ----
with tab4:
    st.subheader("显微组织分类 (Microstructure Classification)")
    with st.spinner("组织分类中..."):
        structure_result = structure_classifier.classify(image)

    pred_class = structure_result.get("predicted_class", "N/A")
    st.markdown(
        f"<div style='text-align:center;padding:30px;background:#f8f9fa;border-radius:12px;'>"
        f"<span style='font-size:14px;color:#7f8c8d;'>分类结果 ({structure_result.get('method', '')})</span><br>"
        f"<span style='font-size:42px;font-weight:bold;color:#2c3e50;'>{pred_class.upper()}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    probs = structure_result.get("probabilities", {})
    if probs:
        st.markdown("### 各类别概率")

        import plotly.graph_objects as go
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        labels = [x[0] for x in sorted_items]
        values = [x[1] * 100 for x in sorted_items]
        colors = ["#e74c3c" if x[0] == pred_class else "#bdc3c7" for x in sorted_items]

        fig = go.Figure(data=[
            go.Bar(x=labels, y=values, marker_color=colors,
                   text=[f"{v:.1f}%" for v in values], textposition="outside")
        ])
        fig.update_layout(
            title="类别概率 (%)", yaxis_title="Probability (%)",
            yaxis_range=[0, max(values) * 1.2], height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    features = structure_result.get("features", {})
    if features:
        with st.expander("纹理特征值", expanded=False):
            feat_df = {"特征": list(features.keys()), "值": [f"{v:.4f}" for v in features.values()]}
            st.dataframe(feat_df, use_container_width=True)

# ---- Tab 5: 综合报告 ----
with tab5:
    st.subheader("综合分析报告")
    st.caption("汇集所有分析结果, 生成 HTML 报告和 JSON 数据")

    metadata = {
        "Filename": uploaded_file.name,
        "Image Size": f"{image.shape[1]} × {image.shape[0]} px",
        "Pixel Scale": f"{pixel_scale} μm/px",
        "Preprocessing": f"Denoise={apply_denoise} ({denoise_strength}), CLAHE={apply_equalize}",
        "Analysis Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    report = MicrographReport()
    html_path = report.generate(
        original_image=image,
        phase_result=phase_result,
        grain_result=grain_result,
        defect_result=defect_result,
        structure_result=structure_result,
        metadata=metadata,
    )
    json_path = report.export_json(
        phase_result=phase_result,
        grain_result=grain_result,
        defect_result=defect_result,
        structure_result=structure_result,
        metadata=metadata,
    )

    st.success(f"报告已生成!")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.download_button(
                label="📥 下载 HTML 报告",
                data=html_content,
                file_name=os.path.basename(html_path),
                mime="text/html",
                use_container_width=True,
            )

    with col_dl2:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                json_content = f.read()
            st.download_button(
                label="📥 下载 JSON 数据",
                data=json_content,
                file_name=os.path.basename(json_path),
                mime="application/json",
                use_container_width=True,
            )

    # 报告预览
    with st.expander("HTML 报告预览", expanded=False):
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)
