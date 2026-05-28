"""模块5: 生成式AI晶体结构发现 — Streamlit 页面"""

import streamlit as st
import os
import sys
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.crystal_generation.crystal_representation import (
    CrystalStructure, ATOMIC_NUM_TO_ELEM, ELEM_TO_ATOMIC_NUM,
)
from modules.crystal_generation.space_group_utils import (
    SPACE_GROUP_NAMES, CRYSTAL_SYSTEMS,
    get_crystal_system, get_lattice_constraints,
    generate_random_structure,
)
from modules.crystal_generation.cgcnn_proxy import DefaultEnergyPredictor
from modules.crystal_generation.validity_checker import StructureValidator
from modules.crystal_generation.structure_generator import CrystalGenerator
import config

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

st.set_page_config(page_title="晶体生成", page_icon="💎", layout="wide")

st.title("💎 生成式AI晶体结构发现")
st.markdown("扩散模型 · CGCNN筛选 · 物理有效性验证 · CIF导出 · DFT对接")


# ---- 侧边栏 ----
with st.sidebar:
    st.header("⚙️ 生成参数")

    st.subheader("目标化学组成")
    elements_input = st.text_input(
        "元素 (逗号分隔)", value="Li,Co,O",
        help="组成晶体的化学元素, 如: Li,Co,O"
    )

    stoich_input = st.text_input(
        "化学计量比 (逗号分隔)", value="1,1,2",
        help="与元素对应的化学计量比例, 如: 1,1,2 表示 LiCoO2"
    )

    st.subheader("空间群")
    sg_options = ["自动 (常见)"] + [f"{sg}: {SPACE_GROUP_NAMES.get(sg, '?')}" for sg in
                                     [1, 2, 14, 15, 62, 139, 166, 194, 221, 225, 227, 229]]
    sg_selection = st.selectbox("目标空间群", sg_options, index=0)

    space_group = 1
    if sg_selection != "自动 (常见)":
        space_group = int(sg_selection.split(":")[0])

    st.subheader("采样参数")
    num_candidates = st.slider("候选数量", 10, 500, 100, 10,
                                help="生成的初始候选结构数")
    num_steps = st.slider("扩散去噪步数", 20, 1000, 100, 10,
                           help="反向扩散采样步数 (越多越精细)")
    temperature = st.slider("采样温度", 0.1, 2.0, 1.0, 0.1,
                             help="越高多样性越大, 越低越保守")
    top_k = st.slider("最终输出数", 1, 50, 10, 1,
                       help="筛选后保留的Top-K候选结构")

    st.subheader("筛选阈值")
    min_stability = st.slider("最低稳定性分数", 0.0, 1.0, 0.3, 0.05)
    max_formation_energy = st.number_input("最高形成能 (eV/atom)", value=0.0, step=0.1)

    st.markdown("---")
    export_cif = st.checkbox("导出CIF文件", value=True)

    st.caption(f"扩散模型: `{config.CRYSTAL_DIFFUSION_PATH}`")
    st.caption(f"CGCNN代理: `{config.CGCNN_PROXY_PATH}`")

# ---- 加载模型 ----
@st.cache_resource
def load_models():
    diffusion = None
    proxy = None

    if HAS_TORCH and os.path.exists(config.CRYSTAL_DIFFUSION_PATH):
        try:
            from modules.crystal_generation.diffusion_model import CrystalDiffusion
            diffusion = CrystalDiffusion(
                hidden_dim=config.DIFFUSION_HIDDEN_DIM,
                num_layers=config.DIFFUSION_NUM_LAYERS,
            )
            ckpt = torch.load(config.CRYSTAL_DIFFUSION_PATH, map_location="cpu")
            diffusion.load_state_dict(
                ckpt.get("model_state_dict", ckpt), strict=False,
            )
            diffusion.eval()
        except Exception as e:
            st.sidebar.warning(f"扩散模型加载失败: {e}")

    if HAS_TORCH and os.path.exists(config.CGCNN_PROXY_PATH):
        try:
            from modules.crystal_generation.cgcnn_proxy import CGCNNProxy
            proxy = CGCNNProxy.load(config.CGCNN_PROXY_PATH)
        except Exception:
            pass

    if proxy is None:
        proxy = DefaultEnergyPredictor()

    validator = StructureValidator()
    generator = CrystalGenerator(
        diffusion_model=diffusion,
        proxy_model=proxy,
        validator=validator,
    )
    return generator, diffusion is not None

generator, has_diffusion = load_models()

# ---- 主界面 ----
st.markdown("---")

col_info, col_status = st.columns([2, 1])
with col_info:
    st.markdown(f"""
    ### 生成管线状态
    - **扩散模型**: {'✅ 已加载' if has_diffusion else '⚠️ 未训练 (使用PyXtal随机生成)'}
    - **CGCNN代理**: {'✅ 已加载' if generator.proxy is not None else '⚠️ 使用经验规则'}
    - **有效性验证**: ✅ 已就绪 (原子间距/配位数/体积/电荷)
    """)

with col_status:
    st.markdown("### 当前元素属性")
    try:
        elements = [e.strip() for e in elements_input.split(",") if e.strip()]
        for el in elements:
            z = ELEM_TO_ATOMIC_NUM.get(el, 0)
            radius = StructureValidator.COVALENT_RADII.get(z, "?")
            system = get_crystal_system(space_group)
            st.caption(f"**{el}**: Z={z}, 半径={radius}Å")
        st.caption(f"**晶系**: {system}")
    except Exception:
        pass

# ---- 运行生成 ----
st.markdown("---")
gen_col, _ = st.columns([1, 3])
with gen_col:
    run_gen = st.button("🚀 开始生成晶体结构", type="primary", use_container_width=True)

if not run_gen:
    st.info("👆 配置参数后点击 **开始生成晶体结构**")
    st.markdown("---")
    st.markdown("""
    ### 使用说明

    1. **输入目标化学组成**: 如 Li,Co,O 和 1,1,2 (LiCoO₂)
    2. **选择空间群**: 目标晶体对称性 (不确定则选"自动")
    3. **调整采样参数**: 候选数量决定搜索广度, 扩散步数决定质量
    4. **运行生成**: 扩散模型采样 → CGCNN快筛 → 物理验证 → Top-K输出
    5. **下载CIF**: 导出候选结构用于DFT验证

    ### 工作流
    ```
    [组成+空间群] → [扩散模型采样] → [CGCNN预测性质] → [物理过滤] → [CIF导出] → [DFT验证]
    ```
    """)
    st.stop()

# ---- 解析输入 ----
elements = [e.strip() for e in elements_input.split(",") if e.strip()]
try:
    stoichiometry = [float(s.strip()) for s in stoich_input.split(",") if s.strip()]
except ValueError:
    st.error("化学计量比格式错误 (应为数字)")
    st.stop()

if len(elements) < 1:
    st.error("请至少输入一种元素")
    st.stop()
if len(stoichiometry) != len(elements):
    st.warning(f"元素({len(elements)})和计量比({len(stoichiometry)})数量不匹配, 已自动调整为等比例")
    stoichiometry = [1] * len(elements)

# ---- 执行生成 ----
output_dir = None
if export_cif:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.DATA_DIR, "generated_structures", timestamp)

with st.spinner(f"正在生成 {num_candidates} 个候选结构..."):
    result = generator.generate(
        elements=elements,
        stoichiometry=stoichiometry,
        space_group=space_group,
        num_candidates=num_candidates,
        num_steps=num_steps,
        temperature=temperature,
        top_k=top_k,
        output_dir=output_dir,
    )

st.success(f"生成完成: {result['num_passed']} / {result['num_total_generated']} 候选通过筛选")

# ============================================================
# 结果展示
# ============================================================
st.markdown("---")
st.subheader("🏆 Top 候选晶体结构")

if result["num_passed"] == 0:
    st.warning("没有结构通过所有有效性检查。请尝试调整参数 (降低温度、更换空间群、增加候选数)。")
    st.stop()

tabs = st.tabs([f"#{i+1}" for i in range(len(result["candidates"]))])

for idx, tab in enumerate(tabs):
    with tab:
        struct = result["candidates"][idx]
        pred = result["predictions"][idx]

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown(f"### {struct._formula_string()}")
            st.markdown(f"**空间群**: {space_group} ({SPACE_GROUP_NAMES.get(space_group, '?')})")
            st.markdown(f"**原子数**: {struct.num_atoms}")

            st.markdown("**晶格参数**:")
            lengths = struct.lattice_lengths
            angles = struct.lattice_angles
            st.write(f"a = {lengths[0]:.4f} Å, b = {lengths[1]:.4f} Å, c = {lengths[2]:.4f} Å")
            st.write(f"α = {angles[0]:.2f}°, β = {angles[1]:.2f}°, γ = {angles[2]:.2f}°")
            st.write(f"体积 = {struct.volume:.2f} Å³, 密度 = {struct.density:.2f} g/cm³")

        with col_b:
            st.markdown("### 预测性质")
            stability = pred.get("stability_score", 0)
            color = "green" if stability > 0.7 else ("orange" if stability > 0.4 else "red")
            st.markdown(f"**稳定性分数**: <span style='color:{color};font-size:24px;'>{stability:.4f}</span>",
                        unsafe_allow_html=True)
            st.metric("形成能 (eV/atom)", f"{pred.get('formation_energy_eV', 0):.4f}")
            st.metric("带隙 (eV)", f"{pred.get('band_gap_eV', 0):.4f}")
            st.metric("是否稳定", "✅ 是" if pred.get("is_stable", False) else "❌ 否")

        # CIF 内容展示
        with st.expander("CIF 文件内容", expanded=False):
            st.code(struct.to_cif_string(), language="text")

        # POSCAR
        with st.expander("VASP POSCAR", expanded=False):
            st.code(generator._to_poscar_string(struct), language="text")

# ---- 汇总表 ----
st.markdown("---")
st.subheader("📊 候选结构汇总")

import pandas as pd
summary_data = []
for i, (struct, pred) in enumerate(zip(result["candidates"], result["predictions"])):
    summary_data.append({
        "排名": i + 1,
        "化学式": struct._formula_string(),
        "空间群": space_group,
        "原子数": struct.num_atoms,
        "体积 (Å³)": f"{struct.volume:.1f}",
        "密度 (g/cm³)": f"{struct.density:.2f}",
        "形成能 (eV/atom)": pred.get("formation_energy_eV", 0),
        "带隙 (eV)": pred.get("band_gap_eV", 0),
        "稳定性": f"{pred.get('stability_score', 0):.4f}",
    })
st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ---- CIF 下载 ----
if result["cif_files"]:
    st.markdown("---")
    st.subheader("📥 下载候选结构")

    import zipfile
    from io import BytesIO

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for cif_path in result["cif_files"]:
            if os.path.exists(cif_path):
                zf.write(cif_path, os.path.basename(cif_path))

        # 添加汇总文件
        summary_lines = [
            "# Crystal Structure Generation Summary",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Elements: {', '.join(elements)}",
            f"# Stoichiometry: {', '.join(str(s) for s in stoichiometry)}",
            f"# Space Group: {space_group}",
            f"# Candidates passed: {result['num_passed']} / {result['num_total_generated']}",
            "",
        ]
        for i, pred in enumerate(result["predictions"]):
            summary_lines.append(
                f"Candidate #{i+1}: E_form={pred['formation_energy_eV']:.4f} eV/atom, "
                f"Gap={pred['band_gap_eV']:.4f} eV, "
                f"Stability={pred['stability_score']:.4f}"
            )
        zf.writestr("README.txt", "\n".join(summary_lines))

    st.download_button(
        label="📥 下载全部CIF (ZIP)",
        data=zip_buf.getvalue(),
        file_name=f"crystal_candidates_{timestamp}.zip",
        mime="application/zip",
        use_container_width=True,
    )

# ---- DFT 工作流 ----
st.markdown("---")
st.subheader("🔬 DFT验证工作流")
st.markdown(f"""
### 后续步骤

**步骤 1: 结构弛豫** (每个候选 ~1-5 CPU小时)
```bash
# INCAR (VASP)
ISIF = 3          # 同时优化原子位置+晶格
IBRION = 2        # 共轭梯度算法
ENCUT = 520       # 截断能 (取决于元素)
EDIFF = 1E-6      # 电子收敛精度
```

**步骤 2: 静态计算** (~0.5 CPU小时/候选)
```bash
ISIF = 0          # 固定晶格
IBRION = -1       # 不弛豫
NSW = 0           # 单点能
```

**步骤 3: 声子谱验证** (Phonopy, ~10 CPU小时/候选)
```bash
# 检查虚频 → 动力学稳定性
phonopy -d --dim="2 2 2" -c POSCAR
```

**步骤 4: 相图分析** (pymatgen)
```python
from pymatgen.analysis.phase_diagram import PhaseDiagram
# 确认候选相在凸包上或接近凸包
```

**步骤 5: 实验合成建议**
- 检查前驱体可用性
- 估算合成温度/压力窗口
- 预测XRD图谱 (pymatgen XRDCalculator)

> 💡 建议先用低精度设置快速筛选 (ENCUT=400, KSPACING=0.5),
> 仅对Top-1~3候选做高精度计算。
""")

st.markdown("---")
st.caption("模块5 · 生成式AI + DFT · 材料基因组计划风格高通量筛选")
