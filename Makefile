# ═══════════════════════════════════════════════════════════
#  SafEye — Project Makefile
#  Common commands for development, testing, and deployment
# ═══════════════════════════════════════════════════════════

.PHONY: help install dev test lint docker-build docker-up docker-down clean models pre-commit pre-commit-run

# Default target
help: ## Show this help message
	@echo "SafEye v3.3 — Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ─── Setup ───

install: ## Install Python dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

models: ## Download AI models (image, audio, text)
	cd models && python download_models.py

# ─── Development ───

dev: ## Run Flask development server
	FLASK_ENV=development python app.py

prod: ## Run with Gunicorn (production)
	gunicorn -c gunicorn.conf.py app:app

# ─── Testing ───

test: ## Run all tests with pytest
	FLASK_ENV=testing pytest -v

test-cov: ## Run tests with coverage report
	FLASK_ENV=testing pytest --cov=backend --cov-report=term-missing -v

lint: ## Run flake8 linter on backend code
	flake8 backend/ --max-line-length=120 --max-complexity=15 --statistics

pre-commit: ## Install pre-commit hooks
	pip install pre-commit
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

# ─── Docker ───

docker-build: ## Build Docker image
	docker compose build

docker-up: ## Start all services
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-logs: ## Tail container logs
	docker compose logs -f --tail=100

# ─── Utilities ───

clean: ## Remove temp files, caches, and uploads
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	rm -f uploads/* 2>/dev/null || true

health: ## Check if the server is healthy
	@curl -s http://localhost:7860/api/health | python -m json.tool

version: ## Show server version info
	@curl -s http://localhost:7860/api/version | python -m json.tool
