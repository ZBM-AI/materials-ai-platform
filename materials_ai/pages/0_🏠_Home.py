"""首页 — 材料科学AI平台总览"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(page_title="首页", page_icon="🏠", layout="wide")

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

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    ### 🔬 显微图像分析
    材料微观结构图像智能分析:
    - 晶粒尺寸测量
    - 相比例计算
    - 孔隙率分析
    - 图像增强处理

    → 前往 **Microscopy_Analysis** 页面体验
    """)

with col5:
    st.markdown("""
    ### 💎 晶体生成
    AI驱动的晶体结构生成与设计:
    - 扩散模型生成
    - VAE潜空间探索
    - 结构优化
    - CIF文件导出

    → 前往 **Crystal_Generation** 页面体验
    """)

with col6:
    st.markdown("""
    ### 🎓 智能学习助手
    材料科学AI助教系统:
    - RAG教材知识库
    - 概念问答
    - 自动出题测验
    - Python代码辅助
    - 实验参数建议

    → 前往 **Learning_Assistant** 页面体验
    """)

st.markdown("---")
st.markdown("""
### 技术栈

| 模块 | 核心技术 |
|------|----------|
| 文献挖掘 | spaCy + DeepSeek API + 规则匹配NER + 依存分析关系抽取 + RAG检索 |
| 知识图谱 | NetworkX + pyvis 可视化 |
| 性能预测 | scikit-learn (RF/GBR/SVR) + pymatgen 成分解析 |
| 显微分析 | OpenCV + skimage 图像处理 |
| 晶体生成 | VAE + 扩散模型 |
| 学习助手 | RAG + DeepSeek LLM + Chroma向量检索 |

### 数据来源
- 文献样本: 材料科学各领域的代表性论文
- 属性数据: 来自 Materials Project 的材料数据
- 知识图谱: 手工策划的种子知识库
""")
