# Streamlit Cloud 入口 — 导入并运行 materials_ai 主应用
import sys
import os

# 确保 materials_ai 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "materials_ai"))

# 导入主 app
from app import *  # noqa: F403 E402
