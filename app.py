"""材料科学AI平台 — HuggingFace Spaces / Streamlit Cloud 入口"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "materials_ai"))

import streamlit as st

st.set_page_config(
    page_title="材料科学AI平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("materials_ai/pages/0_🏠_Home.py"),
    st.Page("materials_ai/pages/1_📚_Literature_Mining.py"),
    st.Page("materials_ai/pages/2_🕸️_Knowledge_Graph.py"),
    st.Page("materials_ai/pages/3_🔮_Property_Prediction.py"),
    st.Page("materials_ai/pages/4_🔬_Microscopy_Analysis.py"),
    st.Page("materials_ai/pages/5_💎_Crystal_Generation.py"),
    st.Page("materials_ai/pages/6_🎓_Learning_Assistant.py"),
]

nav = st.navigation(pages)
nav.run()
