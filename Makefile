.PHONY: help install build up down logs test clean

help: ## 显示帮助信息
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## 安装所有依赖
	pip install -r materials_ai/requirements.txt
	pip install -r backend/requirements.txt

build: ## 构建所有Docker镜像
	docker-compose build

up: ## 启动所有服务 (后台)
	docker-compose up -d

down: ## 停止所有服务
	docker-compose down

logs: ## 查看所有服务日志
	docker-compose logs -f

restart: down up ## 重启所有服务

backend: ## 仅启动后端 (开发模式)
	cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

frontend: ## 仅启动前端 (开发模式)
	cd materials_ai && streamlit run app.py --server.port 8501

test: ## 运行测试
	cd backend && python -m pytest tests/ -v

clean: ## 清理临时文件
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf materials_ai/data/faiss_index/* 2>/dev/null || true

db-init: ## 初始化数据库
	docker-compose exec backend python -c "from app.models.user import user_db; user_db.create_user('admin', 'admin123', role='admin'); print('Admin created')"

api-docs: ## 打开API文档
	@echo "Open http://localhost:8000/docs"

ngrok: ## 暴露本地服务到公网 (需要ngrok)
	ngrok http 8501

dev: ## 完整开发环境启动
	docker-compose up -d mongo redis embedding-api
	@echo "后端: http://localhost:8000/docs"
	@echo "前端: http://localhost:8501"
