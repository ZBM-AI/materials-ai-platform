# Materials AI Platform 🧪

**材料科学人工智能统一平台** — 文献挖掘 · 知识图谱 · 性能预测 · 显微分析 · 晶体生成 · 智能学习助手

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-✓-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 项目简介

Materials AI Platform 是一个面向材料科学研究的全栈 AI 平台，集成 **6 大核心模块**，覆盖从文献挖掘到晶体生成再到学习辅助的完整科研流程。

### 核心功能

| 模块 | 功能 | AI 模型 |
|------|------|---------|
| 📚 **文献挖掘** | PDF解析 · SciBERT NER · 关系抽取 · RAG问答 | SciBERT · Chroma · LangChain |
| 🕸️ **知识图谱** | 本体建模 · Neo4j存储 · Cypher查询 · RGCN推理 | RGCN · py2neo |
| 🔮 **性能预测** | 带隙/形成能/力学/热导率/屈服强度预测 | GPR · XGBoost · MEGNet · CGCNN |
| 🔬 **显微分析** | 物相分割 · 晶粒检测 · 缺陷分类 · 组织识别 | U-Net · YOLOv8-seg · GLCM+LBP |
| 💎 **晶体生成** | 扩散模型 · CGCNN筛选 · 物理验证 · DFT对接 | CDVAE/DiffCSP · EGNN · PyXtal |
| 🎓 **学习助手** | 教材RAG · 自动出题 · Python代码 · 实验建议 | FAISS · Sentence-BERT · OpenAI |

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Streamlit)                      │
│               http://localhost:8501                          │
└───────────────────────┬─────────────────────────────────────┘
                        │ REST API (JWT Auth)
┌───────────────────────▼─────────────────────────────────────┐
│                 Backend (FastAPI)                            │
│               http://localhost:8000                          │
│            OpenAPI Docs: /docs                               │
└───────┬───────┬───────┬───────┬───────┬───────┬─────────────┘
        │       │       │       │       │       │
   ┌────▼──┐ ┌─▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──────┐
   │SciBERT│ │RAG │ │CGCNN│ │YOLO │ │U-Net│ │Crystal  │
   │ NER   │ │API │ │Proxy│ │Grain│ │Phase│ │Diffusion│
   │ :8001 │ │:8002│ │:8003│ │:8004│ │:8005│ │:8006    │
   └───────┘ └────┘ └─────┘ └─────┘ └─────┘ └─────────┘
        │       │       │       │       │       │
   ┌────▼──────▼───────▼───────▼───────▼───────▼─────────────┐
   │                 Shared Storage Layer                      │
   │   MongoDB · Redis · FAISS · Chroma · Neo4j (optional)    │
   └──────────────────────────────────────────────────────────┘
```

**设计原则**:
- 每个 AI 模型独立封装为 Docker 微服务，通过 REST API 调用
- FastAPI 后端统一认证 (JWT)、路由、文件上传
- Streamlit 多页前端提供交互式 UI
- 服务间通过 Docker Compose 网络通信

---

## 📁 项目结构 (Monorepo)

```
materials-ai-platform/
├── backend/                        # FastAPI 后端
│   ├── app/
│   │   ├── main.py                 # 应用入口
│   │   ├── core/
│   │   │   ├── config.py           # 环境变量配置 (Pydantic Settings)
│   │   │   ├── security.py         # JWT 认证
│   │   │   └── dependencies.py     # 依赖注入
│   │   ├── api/v1/
│   │   │   ├── router.py           # API 聚合路由
│   │   │   ├── auth.py             # 注册/登录
│   │   │   ├── literature.py       # Module 1: 文献挖掘
│   │   │   ├── knowledge_graph.py  # Module 2: 知识图谱
│   │   │   ├── property.py         # Module 3: 性能预测
│   │   │   ├── microscopy.py       # Module 4: 显微分析
│   │   │   ├── crystal.py          # Module 5: 晶体生成
│   │   │   └── learning.py         # Module 6: 学习助手
│   │   ├── models/user.py         # 用户数据模型 (SQLite)
│   │   └── schemas/               # Pydantic 请求/响应模型
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                       # Streamlit 前端
│   └── Dockerfile
│
├── materials_ai/                   # 核心 AI 代码库
│   ├── app.py                      # Streamlit 主入口
│   ├── config.py                   # 全局配置
│   ├── pages/                      # 6 个模块页面
│   │   ├── 1_📚_Literature_Mining.py
│   │   ├── 2_🕸️_Knowledge_Graph.py
│   │   ├── 3_🔮_Property_Prediction.py
│   │   ├── 4_🔬_Microscopy_Analysis.py
│   │   ├── 5_💎_Crystal_Generation.py
│   │   └── 6_🎓_Learning_Assistant.py
│   ├── modules/                    # AI 模块实现
│   │   ├── nlp_literature_mining/  # Module 1
│   │   ├── knowledge_graph/        # Module 2
│   │   ├── property_prediction/    # Module 3
│   │   ├── microscopy_analysis/    # Module 4
│   │   ├── crystal_generation/     # Module 5
│   │   └── learning_assistant/     # Module 6
│   ├── saved_models/               # 训练好的模型
│   ├── utils/
│   └── scripts/                    # 训练脚本
│
├── services/                       # Docker 微服务
│   ├── scibert-ner/                # SciBERT NER 服务 (:8001)
│   ├── rag-api/                    # RAG 问答服务 (:8002)
│   ├── cgcnn-proxy/                # CGCNN 代理服务 (:8003)
│   ├── yolo-grain/                 # YOLO 晶粒检测 (:8004)
│   ├── unet-phase/                 # U-Net 物相分割 (:8005)
│   ├── crystal-diffusion/          # 晶体扩散模型 (:8006)
│   └── embedding-api/              # 文本 Embedding (:8007)
│
├── docker-compose.yml              # 一键启动所有服务
├── .env.example                    # 环境变量模板
├── Makefile                        # 常用命令
└── README.md                       # 本文件
```

---

## 🚀 快速开始

### 前置要求

- Docker & Docker Compose v2+
- Python 3.11+ (本地开发)
- Git

### 方式一: Docker Compose (推荐)

```bash
# 1. 克隆项目
git clone <repository-url>
cd materials-ai-platform

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env: 设置 SECRET_KEY 和 OPENAI_API_KEY (可选)

# 3. 一键启动所有服务
docker-compose up -d

# 4. 初始化管理员账户
make db-init

# 5. 访问
# 前端: http://localhost:8501
# 后端 API 文档: http://localhost:8000/docs
# 健康检查: http://localhost:8000/health
```

### 方式二: 本地开发

```bash
# 1. 安装依赖
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r materials_ai/requirements.txt
pip install -r backend/requirements.txt

# 2. 启动后端 (终端1)
make backend
# 或: cd backend && uvicorn app.main:app --reload --port 8000

# 3. 启动前端 (终端2)
make frontend
# 或: cd materials_ai && streamlit run app.py --server.port 8501

# 4. 访问
# 前端: http://localhost:8501
# API 文档: http://localhost:8000/docs
```

### 方式三: 仅启动必要的微服务

```bash
# 只启动基础设施 + 你需要的 AI 服务
docker-compose up -d mongo redis embedding-api scibert-ner
make backend
make frontend
```

---

## 📡 API 文档

### OpenAPI (Swagger)

启动后端后访问:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### API 端点总览

| 方法 | 端点 | 描述 | 认证 |
|------|------|------|------|
| `POST` | `/api/v1/auth/register` | 用户注册 | 否 |
| `POST` | `/api/v1/auth/login` | 用户登录 | 否 |
| `POST` | `/api/v1/literature/search` | 搜索论文 | JWT |
| `POST` | `/api/v1/literature/analyze` | SciBERT NER 分析 | JWT |
| `POST` | `/api/v1/literature/extract-triplets` | 关系三元组抽取 | JWT |
| `POST` | `/api/v1/literature/rag` | RAG 文献问答 | JWT |
| `POST` | `/api/v1/kg/query` | 知识图谱查询 | JWT |
| `GET` | `/api/v1/kg/stats` | 图谱统计 | JWT |
| `POST` | `/api/v1/property/predict` | 性能预测 | JWT |
| `GET` | `/api/v1/property/models` | 可用模型列表 | JWT |
| `POST` | `/api/v1/microscopy/analyze` | 显微图像分析 | JWT |
| `POST` | `/api/v1/crystal/generate` | 晶体结构生成 | JWT |
| `POST` | `/api/v1/learning/qa` | 概念问答 | JWT |
| `POST` | `/api/v1/learning/quiz` | 自动出题 | JWT |
| `POST` | `/api/v1/learning/code` | Python 代码生成 | JWT |
| `POST` | `/api/v1/learning/predict-phase` | 相预测 | JWT |
| `POST` | `/api/v1/learning/heat-treatment` | 热处理建议 | JWT |

### 使用示例

```bash
# 1. 注册
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123"}'

# 2. 登录 (获取 Token)
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123"}' | jq -r '.access_token')

# 3. 文献 RAG 问答
curl -X POST http://localhost:8000/api/v1/literature/rag \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question":"What is the Hall-Petch relationship?","k":3}'

# 4. 性能预测
curl -X POST http://localhost:8000/api/v1/property/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"formula":"TiO2","targets":["band_gap","formation_energy"]}'

# 5. 晶体生成
curl -X POST http://localhost:8000/api/v1/crystal/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"elements":["Li","Co","O"],"stoichiometry":[1,1,2],"space_group":166,"num_candidates":50,"top_k":5}'

# 6. 学习助手问答
curl -X POST http://localhost:8000/api/v1/learning/qa \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question":"解释位错攀移机制"}'
```

---

## 🔧 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `SECRET_KEY` | (必填) | JWT 签名密钥 |
| `MONGODB_URI` | `mongodb://mongo:27017` | MongoDB 连接 |
| `REDIS_URL` | `redis://redis:6379/0` | Redis 连接 |
| `OPENAI_API_KEY` | (可选) | OpenAI API Key (LLM 增强) |
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j 连接 (可选) |
| `SCIBERT_NER_URL` | `http://scibert-ner:8001/predict` | NER 服务地址 |
| `RAG_API_URL` | `http://rag-api:8002` | RAG 服务地址 |
| `CGCNN_PROXY_URL` | `http://cgcnn-proxy:8003/predict` | CGCNN 服务地址 |
| `YOLO_GRAIN_URL` | `http://yolo-grain:8004/predict` | YOLO 服务地址 |
| `UNET_PHASE_URL` | `http://unet-phase:8005/predict` | U-Net 服务地址 |
| `CRYSTAL_DIFFUSION_URL` | `http://crystal-diffusion:8006/generate` | 扩散模型地址 |
| `EMBEDDING_API_URL` | `http://embedding-api:8007` | Embedding 服务地址 |

完整配置见 [.env.example](.env.example)

---

## 🐳 Docker 微服务一览

| 服务 | 端口 | 镜像大小 | GPU 建议 |
|------|------|---------|----------|
| `embedding-api` | 8007 | ~500MB | CPU OK |
| `scibert-ner` | 8001 | ~2GB | GPU 推荐 |
| `rag-api` | 8002 | ~1GB | CPU OK |
| `cgcnn-proxy` | 8003 | ~1.5GB | GPU 推荐 |
| `yolo-grain` | 8004 | ~2GB | GPU 推荐 |
| `unet-phase` | 8005 | ~1.5GB | GPU 推荐 |
| `crystal-diffusion` | 8006 | ~2.5GB | GPU 推荐 |
| `backend` | 8000 | ~300MB | CPU OK |
| `frontend` | 8501 | ~500MB | CPU OK |
| `mongo` | 27017 | ~700MB | CPU OK |
| `redis` | 6379 | ~30MB | CPU OK |

---

## 📊 开发路线图

### Phase 1: 核心平台 ✅ (当前)
- [x] 6大AI模块完成
- [x] FastAPI 统一后端 + JWT认证
- [x] Docker 微服务化
- [x] Streamlit 多页前端
- [x] OpenAPI 自动文档

### Phase 2: 生产就绪 (计划中)
- [ ] PostgreSQL / SQLAlchemy 替换 SQLite
- [ ] Celery 异步任务队列 (长时间AI推理)
- [ ] Nginx 反向代理 + HTTPS
- [ ] Kubernetes Helm Chart
- [ ] GitHub Actions CI/CD
- [ ] 模型版本管理 (MLflow / DVC)
- [ ] 用户管理面板

### Phase 3: 高级功能 (计划中)
- [ ] 多租户支持
- [ ] WebSocket 实时推理进度
- [ ] 实验数据自动采集 (ELN 集成)
- [ ] 高通量 DFT 自动提交
- [ ] 移动端适配 (PWA)
- [ ] 多语言支持 (中文/English)

---

## 🧪 测试

```bash
# 运行后端测试
docker-compose exec backend python -m pytest tests/ -v

# 运行单个模块测试
python -m pytest tests/test_literature.py -v

# API 端到端测试
python -m pytest tests/test_api.py -v
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 📝 引用

如果本平台对你的研究有帮助，请引用:

```bibtex
@software{materials_ai_platform,
  title = {Materials AI Platform},
  year = {2025},
  description = {材料科学人工智能统一平台},
  url = {<repository-url>},
}
```

### 参考文献

- **SciBERT**: Beltagy et al., "SciBERT: A Pretrained Language Model for Scientific Text", EMNLP 2019
- **CGCNN**: Xie & Grossman, "Crystal Graph Convolutional Neural Networks", Phys. Rev. Lett. 2018
- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
- **YOLOv8**: Ultralytics, https://github.com/ultralytics/ultralytics
- **CDVAE**: Xie et al., "Crystal Diffusion Variational Autoencoder", ICLR 2022
- **EGNN**: Satorras et al., "E(n) Equivariant Graph Neural Networks", ICML 2021
- **MEGNet**: Chen et al., "Graph Networks as a Universal Machine Learning Framework", Chem. Mater. 2019

---

## 📄 许可证

MIT License

---

**Made with ❤️ for Materials Science**
