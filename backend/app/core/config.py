"""统一配置 — 环境变量 + Pydantic Settings."""

from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # ---- 应用 ----
    APP_NAME: str = "Materials AI Platform"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # ---- 服务器 ----
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:8501", "http://localhost:3000"]

    # ---- 数据库 ----
    DATABASE_URL: str = "sqlite:///./data/users.db"
    MONGODB_URI: str = "mongodb://mongo:27017"
    MONGODB_DB: str = "materials_ai"
    REDIS_URL: str = "redis://redis:6379/0"

    # ---- JWT ----
    SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24h

    # ---- AI 微服务 URLs ----
    SCIBERT_NER_URL: str = "http://scibert-ner:8001/predict"
    RAG_API_URL: str = "http://rag-api:8002"
    CGCNN_PROXY_URL: str = "http://cgcnn-proxy:8003/predict"
    YOLO_GRAIN_URL: str = "http://yolo-grain:8004/predict"
    UNET_PHASE_URL: str = "http://unet-phase:8005/predict"
    CRYSTAL_DIFFUSION_URL: str = "http://crystal-diffusion:8006/generate"
    EMBEDDING_API_URL: str = "http://embedding-api:8007"

    # ---- OpenAI (可选) ----
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ---- 文件存储 ----
    UPLOAD_DIR: str = "./data/uploads"
    MAX_UPLOAD_SIZE_MB: int = 500

    # ---- 模型路径 (本地回退) ----
    SCIBERT_MODEL: str = "allenai/scibert_scivocab_uncased"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()
