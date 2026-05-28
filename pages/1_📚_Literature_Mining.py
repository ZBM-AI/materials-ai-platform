# Bridge: Streamlit Cloud entry → materials_ai module
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "materials_ai"))
exec(open(os.path.join(os.path.dirname(__file__), "..", "materials_ai", "pages", "1_📚_Literature_Mining.py"), encoding="utf-8").read())
