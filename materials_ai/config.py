"""
材料科学AI平台 - 集中配置文件
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, "data")
PAPERS_DIR = os.path.join(DATA_DIR, "papers")
PARSED_DIR = os.path.join(DATA_DIR, "parsed")
KG_DIR = os.path.join(DATA_DIR, "knowledge_graph")
PROPERTY_DATA_DIR = os.path.join(DATA_DIR, "property_data")

# 预解析结果
PARSED_PAPERS_FILE = os.path.join(PARSED_DIR, "parsed_papers.json")

# 知识图谱
SEED_KG_FILE = os.path.join(KG_DIR, "seed_kg.json")

# Neo4j 图数据库 (可选)
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "neo4j123")

# RGCN 链接预测
RGCN_HIDDEN_DIM = int(os.environ.get("RGCN_HIDDEN_DIM", "64"))
RGCN_NUM_LAYERS = int(os.environ.get("RGCN_NUM_LAYERS", "2"))
RGCN_EPOCHS = int(os.environ.get("RGCN_EPOCHS", "100"))
RGCN_LEARNING_RATE = float(os.environ.get("RGCN_LR", "0.01"))
RGCN_MODEL_DIR = os.path.join(DATA_DIR, "models", "rgcn")

# 属性预测数据
BAND_GAP_DATA = os.path.join(PROPERTY_DATA_DIR, "band_gap_data.csv")
FORMATION_ENERGY_DATA = os.path.join(PROPERTY_DATA_DIR, "formation_energy_data.csv")
MECHANICAL_DATA = os.path.join(PROPERTY_DATA_DIR, "mechanical_data.csv")
THERMAL_CONDUCTIVITY_DATA = os.path.join(PROPERTY_DATA_DIR, "thermal_conductivity_data.csv")
YIELD_STRENGTH_DATA = os.path.join(PROPERTY_DATA_DIR, "yield_strength_data.csv")

# 模型保存
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
BAND_GAP_MODEL = os.path.join(SAVED_MODELS_DIR, "band_gap_model.pkl")
FORMATION_ENERGY_MODEL = os.path.join(SAVED_MODELS_DIR, "formation_energy_model.pkl")
MECHANICAL_MODEL = os.path.join(SAVED_MODELS_DIR, "mechanical_model.pkl")
THERMAL_CONDUCTIVITY_MODEL = os.path.join(SAVED_MODELS_DIR, "thermal_conductivity_model.pkl")
YIELD_STRENGTH_MODEL = os.path.join(SAVED_MODELS_DIR, "yield_strength_model.pkl")

# spaCy模型
SPACY_MODEL = "en_core_web_sm"

# 实体/关系类型
ENTITY_TYPES = ["material", "property", "processing_method", "crystal_structure",
                "application", "property_value", "synthesis_method", "microstructure"]
RELATION_TYPES = ["hasProperty", "processedBy", "hasStructure", "usedIn",
                  "hasValue", "synthesizedBy", "hasMicrostructure", "relatedTo"]

ENTITY_COLORS = {
    "material": "#FF6B6B",          # 红色
    "property": "#4ECDC4",          # 青色
    "processing_method": "#45B7D1", # 蓝色
    "crystal_structure": "#96CEB4",  # 绿色
    "application": "#FFEAA7",       # 黄色
    "property_value": "#DDA0DD",    # 紫色
    "synthesis_method": "#F39C12",  # 橙色
    "microstructure": "#8E44AD",    # 深紫
}

# ============================================================
# Module 1: 文献挖掘升级 — 路径配置
# ============================================================
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
NER_TRAINING_DIR = os.path.join(DATA_DIR, "ner_training")
NER_MODEL_DIR = os.path.join(SAVED_MODELS_DIR, "scibert_ner")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "downloaded_papers")
TRIPLETS_FILE = os.path.join(PARSED_DIR, "triplets.json")
MONGODB_COLLECTION = "materials_triplets"

# SciBERT NER 配置
SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
SCIBERT_MAX_LENGTH = 512
SCIBERT_BATCH_SIZE = 8
SCIBERT_LEARNING_RATE = 2e-5
SCIBERT_NUM_EPOCHS = 3

# BIO标签 -> ID 映射
BIO_TAGS = ["O", "B-MAT", "I-MAT", "B-SYN", "I-SYN", "B-PRO", "I-PRO",
            "B-VAL", "I-VAL", "B-MIC", "I-MIC", "B-APP", "I-APP"]
TAG_TO_ID = {tag: i for i, tag in enumerate(BIO_TAGS)}
ID_TO_TAG = {i: tag for tag, i in TAG_TO_ID.items()}

# BIO前缀 -> 实体类型
BIO_PREFIX_TO_TYPE = {
    "MAT": "material",
    "SYN": "synthesis_method",
    "PRO": "property",
    "VAL": "property_value",
    "MIC": "microstructure",
    "APP": "application",
}

# RAG / Embedding 配置
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "materials_papers"
RAG_TOP_K = 5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# 下载器配置
MAX_PAPERS_PER_QUERY = 50

# MongoDB (可选)
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = "materials_ai"

# ============================================================
# Module 4: 微观图像智能分析 — 路径配置
# ============================================================
MICROSCOPY_MODELS_DIR = os.path.join(SAVED_MODELS_DIR, "microscopy")
UNET_MODEL_PATH = os.path.join(MICROSCOPY_MODELS_DIR, "unet_phase.pt")
YOLO_GRAIN_MODEL_PATH = os.path.join(MICROSCOPY_MODELS_DIR, "yolov8_grain.pt")
DEFECT_CNN_MODEL_PATH = os.path.join(MICROSCOPY_MODELS_DIR, "defect_cnn.pt")
STRUCTURE_CNN_MODEL_PATH = os.path.join(MICROSCOPY_MODELS_DIR, "structure_cnn.pt")

# U-Net 物相分割配置
UNET_N_CLASSES = 4
UNET_N_CHANNELS = 1
UNET_BATCH_SIZE = 4
UNET_LEARNING_RATE = 1e-4
UNET_NUM_EPOCHS = 100

# YOLOv8 晶粒检测配置
YOLO_IMGSZ = 640
YOLO_EPOCHS = 100
YOLO_BATCH_SIZE = 8

# 图像预处理默认参数
DEFAULT_PIXEL_SCALE_UM = 0.1       # μm/pixel
DEFAULT_MIN_GRAIN_AREA_PX = 100    # 最小晶粒面积 (px)
DEFAULT_MIN_DEFECT_AREA_PX = 20    # 最小缺陷面积 (px)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# 标注数据目录
LABELME_DIR = os.path.join(DATA_DIR, "labelme_annotations")
COCO_ANNOTATIONS_PATH = os.path.join(DATA_DIR, "coco_annotations", "micrographs.json")

# ============================================================
# Module 5: 生成式AI晶体结构发现 — 路径配置
# ============================================================
CRYSTAL_MODELS_DIR = os.path.join(SAVED_MODELS_DIR, "crystal_generation")
CRYSTAL_DIFFUSION_PATH = os.path.join(CRYSTAL_MODELS_DIR, "crystal_diffusion.pt")
CGCNN_PROXY_PATH = os.path.join(CRYSTAL_MODELS_DIR, "cgcnn_proxy.pt")
GENERATED_STRUCTURES_DIR = os.path.join(DATA_DIR, "generated_structures")

# 扩散模型配置
DIFFUSION_HIDDEN_DIM = int(os.environ.get("DIFFUSION_HIDDEN_DIM", "128"))
DIFFUSION_NUM_LAYERS = int(os.environ.get("DIFFUSION_NUM_LAYERS", "4"))
DIFFUSION_NUM_TIMESTEPS = int(os.environ.get("DIFFUSION_NUM_TIMESTEPS", "1000"))
DIFFUSION_LEARNING_RATE = float(os.environ.get("DIFFUSION_LR", "1e-4"))
DIFFUSION_BATCH_SIZE = int(os.environ.get("DIFFUSION_BATCH_SIZE", "32"))
DIFFUSION_NUM_EPOCHS = int(os.environ.get("DIFFUSION_NUM_EPOCHS", "200"))

# CGCNN代理模型配置
CGCNN_HIDDEN_DIM = int(os.environ.get("CGCNN_HIDDEN_DIM", "128"))
CGCNN_NUM_LAYERS = int(os.environ.get("CGCNN_NUM_LAYERS", "3"))

# 结构有效性验证默认阈值
MIN_DISTANCE_RATIO = float(os.environ.get("MIN_DISTANCE_RATIO", "0.65"))
MAX_COORDINATION = int(os.environ.get("MAX_COORDINATION", "16"))
MIN_VOLUME_PER_ATOM = float(os.environ.get("MIN_VOLUME_PER_ATOM", "5.0"))
MAX_VOLUME_PER_ATOM = float(os.environ.get("MAX_VOLUME_PER_ATOM", "50.0"))

# DFT对接默认配置
DEFAULT_ENCUT = 520
DEFAULT_KSPACING = 0.3

# ============================================================
# Module 6: 智能学习助手 — 路径配置
# ============================================================
TEXTBOOKS_DIR = os.path.join(DATA_DIR, "textbooks")
FAISS_INDEX_PATH = os.path.join(os.path.expanduser("~"), ".materials_ai", "faiss_index")  # 避免中文路径(FAISS C++不支持)

# RAG 配置
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 100
RAG_RETRIEVAL_K = 5

# DeepSeek V4 Pro API — LLM增强 (兼容 OpenAI SDK)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_MAX_TOKENS = int(os.environ.get("DEEPSEEK_MAX_TOKENS", "4096"))

# 兼容旧变量名
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", DEEPSEEK_API_KEY)
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", DEEPSEEK_MODEL)
