"""材料科学AI平台 — Streamlit 主入口"""

import streamlit as st

st.set_page_config(
    page_title="材料科学AI平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🔬 材料科学AI平台")
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 三大功能模块

- **📚 文献挖掘**: 从材料科学论文中自动提取实体和关系
- **🕸️ 知识图谱**: 构建和查询材料知识网络
- **🔮 性能预测**: 基于化学成分预测材料性能
""")

st.sidebar.markdown("---")
st.sidebar.info("""
**使用说明**: 在左侧导航中选择对应功能页面。
每个模块均可独立运行和演示。
""")

st.title("🔬 材料科学人工智能平台")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📚 文献挖掘
    利用自然语言处理技术从材料科学文献中自动提取:
    - 材料名称与化学式
    - 性能参数与数值
    - 加工方法与工艺
    - 晶体结构类型
    - 应用领域

    → 前往 **Literature_Mining** 页面体验
    """)

with col2:
    st.markdown("""
    ### 🕸️ 知识图谱
    构建材料科学知识网络，实现:
    - 材料-性能-工艺-结构关联
    - 交互式图谱可视化
    - 实体搜索与路径发现
    - 种子知识图谱预加载

    → 前往 **Knowledge_Graph** 页面体验
    """)

with col3:
    st.markdown("""
    ### 🔮 性能预测
    基于化学成分预测材料性能:
    - 带隙 (Band Gap)
    - 形成能 (Formation Energy)
    - 体模量 (Bulk Modulus)
    - 特征重要性分析

    → 前往 **Property_Prediction** 页面体验
    """)

st.markdown("---")
st.markdown("""
### 技术栈

| 模块 | 核心技术 |
|------|----------|
| 文献挖掘 | spaCy + pdfplumber + 规则匹配NER + 依存分析关系抽取 |
| 知识图谱 | NetworkX + pyvis 可视化 |
| 性能预测 | scikit-learn (RF/GBR/SVR) + pymatgen 成分解析 |

### 数据来源
- 文献样本: 材料科学各领域的5篇代表性论文摘要
- 属性数据: 来自 Materials Project 的 ~300 种材料数据
- 知识图谱: 手工策划的 ~50 实体 / ~80 关系种子知识库
""")
